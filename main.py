from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List, Optional, Union
from transformers import BertTokenizer, BertForMaskedLM, BertForSequenceClassification, BertForTokenClassification, BertForQuestionAnswering
import torch
from datetime import datetime
import argparse
import os
import asyncio
from enum import Enum

class TaskType(Enum):
    MASK_FILL = "mask_fill"
    CLASSIFICATION = "classification"
    NER = "ner"
    QA = "qa"

# 解析命令行参数
parser = argparse.ArgumentParser(description='BART推理服务')
parser.add_argument('--service-name', type=str, default='bart', help='服务名称')
parser.add_argument('--model-path', type=str, required=True, help='模型目录路径')
parser.add_argument('--host', type=str, default='0.0.0.0', help='服务主机地址')
parser.add_argument('--port', type=int, default=8000, help='服务端口号')

args = parser.parse_args()

app = FastAPI()

# 全局变量存储模型和tokenizer
models = {}
tokenizer = None
task_type = None

# 创建信号量来控制并发请求数量
semaphore = asyncio.Semaphore(5)  # 限制最大并发请求数为5

# 在启动时加载模型
def load_model():
    global models, tokenizer
    try:
        model_path = args.model_path
        if not os.path.exists(model_path):
            print(f"错误：模型路径不存在: {model_path}")
            return False

        # 加载tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_path, use_fast=True)
        
        # 加载不同任务的模型
        models[TaskType.MASK_FILL] = BertForMaskedLM.from_pretrained(model_path)
        models[TaskType.CLASSIFICATION] = BertForSequenceClassification.from_pretrained(model_path)
        models[TaskType.NER] = BertForTokenClassification.from_pretrained(model_path)
        models[TaskType.QA] = BertForQuestionAnswering.from_pretrained(model_path)
        
        # 将所有模型设置为评估模式
        for model in models.values():
            model.eval()
            
        print("所有模型加载成功！")
        return True
    except Exception as e:
        print(f"错误：模型加载失败: {str(e)}")
        return False

class BaseRequest(BaseModel):
    model: str
    max_tokens: Optional[int] = 50
    temperature: Optional[float] = 1.0

class CompletionRequest(BaseRequest):
    texts: Union[str, List[str]]

    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        if isinstance(v, str):
            return [v]
        return v

class ClassificationRequest(BaseRequest):
    texts: Union[str, List[str]]
    labels: Optional[List[str]] = None

    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        if isinstance(v, str):
            return [v]
        return v

class NERRequest(BaseRequest):
    texts: Union[str, List[str]]

    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        if isinstance(v, str):
            return [v]
        return v

class QARequest(BaseRequest):
    questions: Union[str, List[str]]
    contexts: Union[str, List[str]]

    @field_validator('questions', 'contexts')
    @classmethod
    def validate_lists(cls, v):
        if isinstance(v, str):
            return [v]
        return v

class BaseResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str

class CompletionResponse(BaseResponse):
    choices: List[dict]

class ClassificationResponse(BaseResponse):
    predictions: List[dict]

class NERResponse(BaseResponse):
    entities: List[dict]

class QAResponse(BaseResponse):
    answers: List[dict]

@app.post("/v1/bart/mask_fill")
async def create_mask_fill(request: CompletionRequest):
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
                outputs = models[TaskType.MASK_FILL](inputs["input_ids"], attention_mask=inputs["attention_mask"])

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

@app.post("/v1/bart/classification")
async def create_classification(request: ClassificationRequest):
    async with semaphore:
        if request.model != args.service_name:
            raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
        
        try:
            # 批量处理输入文本
            inputs = tokenizer(request.texts, 
                              return_tensors="pt", 
                              padding=True, 
                              truncation=True, 
                              max_length=request.max_tokens)
            
            with torch.no_grad():
                outputs = models[TaskType.CLASSIFICATION](**inputs)
            
            # 获取预测结果
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(predictions, dim=-1)
            
            # 如果提供了标签列表，使用提供的标签
            labels = request.labels if request.labels else [str(i) for i in range(predictions.shape[1])]
            
            # 构建返回结果
            predictions_list = [
                {
                    "label": labels[pred],
                    "score": float(probs[pred]),
                    "index": i
                }
                for i, (pred, probs) in enumerate(zip(predicted_labels, predictions))
            ]
            
            response = ClassificationResponse(
                id=f"cls-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                object="text_classification",
                created=int(datetime.now().timestamp()),
                model=request.model,
                predictions=predictions_list
            )
            
            return response
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/v1/bart/ner")
async def create_ner(request: NERRequest):
    async with semaphore:
        if request.model != args.service_name:
            raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
        
        try:
            # 批量处理输入文本
            inputs = tokenizer(request.texts, 
                              return_tensors="pt", 
                              padding=True, 
                              truncation=True, 
                              max_length=request.max_tokens)
            
            with torch.no_grad():
                outputs = models[TaskType.NER](**inputs)
            
            # 获取预测结果
            predictions = torch.softmax(outputs.logits, dim=-1)
            pred_labels = torch.argmax(predictions, dim=-1)
            
            # 定义NER标签映射
            ner_labels = {
                1: "PER",  # 人名
                2: "ORG",  # 组织
                3: "LOC",  # 地点
                4: "TIME", # 时间
                5: "MISC"  # 其他
            }
            
            # 处理每个文本的实体
            entities_list = []
            for i, (input_ids, pred_label_ids, pred_probs) in enumerate(zip(inputs["input_ids"], pred_labels, predictions)):
                # 解码token并移除特殊token
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                text = request.texts[i]
                current_entity = None
                entities = []
                
                for j, (token, label_id, probs) in enumerate(zip(tokens, pred_label_ids, pred_probs)):
                    # 跳过特殊token
                    if token in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]:
                        continue
                        
                    # 移除##前缀
                    if token.startswith("##"):
                        token = token[2:]
                    
                    label_id = label_id.item()
                    if label_id == 0:  # O标签
                        if current_entity:
                            # 计算在原始文本中的位置
                            start_pos = text.find(current_entity["text"])
                            if start_pos != -1:
                                current_entity["start"] = start_pos
                                current_entity["end"] = start_pos + len(current_entity["text"])
                                entities.append(current_entity)
                            current_entity = None
                    else:
                        label = ner_labels.get(label_id, "MISC")
                        confidence = float(probs[label_id])
                        
                        if current_entity is None:
                            current_entity = {
                                "text": token,
                                "label": label,
                                "confidence": confidence
                            }
                        else:
                            # 如果标签相同，则合并实体
                            if current_entity["label"] == label:
                                current_entity["text"] += token
                                current_entity["confidence"] = max(current_entity["confidence"], confidence)
                            else:
                                # 如果标签不同，保存当前实体并开始新实体
                                start_pos = text.find(current_entity["text"])
                                if start_pos != -1:
                                    current_entity["start"] = start_pos
                                    current_entity["end"] = start_pos + len(current_entity["text"])
                                    entities.append(current_entity)
                                current_entity = {
                                    "text": token,
                                    "label": label,
                                    "confidence": confidence
                                }
                
                if current_entity:
                    start_pos = text.find(current_entity["text"])
                    if start_pos != -1:
                        current_entity["start"] = start_pos
                        current_entity["end"] = start_pos + len(current_entity["text"])
                        entities.append(current_entity)
                
                entities_list.append({
                    "text": text,
                    "entities": entities,
                    "index": i
                })
            
            response = NERResponse(
                id=f"ner-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                object="named_entity_recognition",
                created=int(datetime.now().timestamp()),
                model=request.model,
                entities=entities_list
            )
            
            return response
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/v1/bart/qa")
async def create_qa(request: QARequest):
    async with semaphore:
        if request.model != args.service_name:
            raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
        
        try:
            answers_list = []
            
            # 处理每个问题和上下文对
            for i, (question, context) in enumerate(zip(request.questions, request.contexts)):
                # 准备输入
                inputs = tokenizer(
                    question,
                    context,
                    return_tensors="pt",
                    max_length=request.max_tokens,
                    truncation=True,
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = models[TaskType.QA](**inputs)
                
                # 获取开始和结束位置的预测
                start_scores = outputs.start_logits
                end_scores = outputs.end_logits
                
                # 找到最可能的答案范围
                start_idx = torch.argmax(start_scores)
                end_idx = torch.argmax(end_scores)
                
                # 确保开始位置在结束位置之前
                if start_idx <= end_idx:
                    answer_tokens = inputs["input_ids"][0][start_idx:end_idx+1]
                    answer_text = tokenizer.decode(answer_tokens)
                else:
                    answer_text = ""
                
                answers_list.append({
                    "question": question,
                    "context": context,
                    "answer": answer_text,
                    "index": i
                })
            
            response = QAResponse(
                id=f"qa-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                object="question_answering",
                created=int(datetime.now().timestamp()),
                model=request.model,
                answers=answers_list
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