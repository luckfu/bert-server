from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List, Optional, Union
from transformers import BertTokenizer, BertForMaskedLM
import torch
from datetime import datetime
import argparse
import os
import asyncio

# 解析命令行参数
parser = argparse.ArgumentParser(description='BART推理服务')
parser.add_argument('--service-name', type=str, default='bart', help='服务名称')
parser.add_argument('--model-path', type=str, required=True, help='模型目录路径')
parser.add_argument('--host', type=str, default='0.0.0.0', help='服务主机地址')
parser.add_argument('--port', type=int, default=8000, help='服务端口号')

args = parser.parse_args()

app = FastAPI()

# 全局变量存储模型和tokenizer
model = None
tokenizer = None

# 创建信号量来控制并发请求数量
semaphore = asyncio.Semaphore(5)  # 限制最大并发请求数为5

# 在启动时加载模型
def load_model():
    global model, tokenizer
    try:
        model_path = args.model_path
        if not os.path.exists(model_path):
            print(f"错误：模型路径不存在: {model_path}")
            return False
        # 使用BertTokenizer和BertForMaskedLM加载模型
        tokenizer = BertTokenizer.from_pretrained(model_path, use_fast=True)
        model = BertForMaskedLM.from_pretrained(model_path)
        model.eval()
        print("模型加载成功！")
        return True
    except Exception as e:
        print(f"错误：模型加载失败: {str(e)}")
        return False

class CompletionRequest(BaseModel):
    model: str
    texts: Union[str, List[str]]
    max_tokens: Optional[int] = 50
    temperature: Optional[float] = 1.0

    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        if isinstance(v, str):
            return [v]
        return v

class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]

@app.post("/v1/bart/completions")
async def create_completion(request: CompletionRequest):
    # 使用信号量控制并发
    async with semaphore:
        # 验证请求的model是否与service-name匹配
        if request.model != args.service_name:
            raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
        
        try:
            # 批量处理输入文本
            inputs = tokenizer(request.texts, 
                              return_tensors="pt", 
                              padding=True, 
                              truncation=True, 
                              max_length=request.max_tokens)
            
            # 找到[MASK]对应的位置
            mask_positions = []
            for input_ids in inputs["input_ids"]:
                mask_pos = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                mask_positions.append(mask_pos)

            with torch.no_grad():
                outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])

            decoded_outputs = []
            for i, (logits, mask_pos) in enumerate(zip(outputs.logits, mask_positions)):
                # 只取[MASK]位置的预测结果
                mask_predictions = logits[mask_pos]
                predicted_tokens = torch.argmax(mask_predictions, dim=-1)
                # 直接解码预测的token
                predicted_text = tokenizer.decode(predicted_tokens, skip_special_tokens=True).strip()
                decoded_outputs.append(predicted_text)
            
            # 构建返回结果
            choices = [
                {
                    "text": output,
                    "index": i,
                    "finish_reason": "length" if len(output) >= request.max_tokens else "stop"
                }
                for i, output in enumerate(decoded_outputs)
            ]
            
            response = CompletionResponse(
                id=f"cmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                object="text_completion",
                created=int(datetime.now().timestamp()),
                model=request.model,
                choices=choices
            )
            
            return response
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    # 在启动服务前加载模型
    if not load_model():
        print("服务启动失败：模型加载错误")
        exit(1)
    
    import uvicorn
    print(f"服务启动成功！访问地址: http://{args.host}:{args.port}/v1/bart/completions，当前使用模型: {args.service_name}")
    uvicorn.run(app, host=args.host, port=args.port)