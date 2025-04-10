# BART 推理服务

这是一个基于FastAPI的BART模型推理服务，模拟了OpenAI API的接口格式。

## 启动服务

服务启动时需要指定以下命令行参数：

- `--service-name`: 服务使用的模型名称，默认为'bart'，请求时需要在model字段中使用相同的值
- `--model-path`: 模型目录路径（必需参数）
- `--host`: 服务主机地址，默认为'0.0.0.0'
- `--port`: 服务端口号，默认为8000

```bash
 python3 bert_server.pyc --model-path ./bert-base-chinese --service-name bert-base-chinese
```

服务将在 http://localhost:8000 启动

## API 使用说明

### 文本补全接口

**Endpoint:** `/v1/bert/mask_fill`

**请求方法:** POST

**请求参数:**

```json
{
    "model": "bert-base-chinese",  // BART模型名称
    "texts": ["需要补全的文本1", "需要补全的文本2"],  // 输入文本列表
    "max_tokens": 50,  // 可选，最大生成长度
    "temperature": 1.0  // 可选，生成多样性参数
}
```

**返回格式:**

```json
{
    "id": "cmpl-20231205123456",
    "object": "text_completion",
    "created": 1701765432,
    "model": "facebook/bart-base",
    "choices": [
        {
            "text": "生成的文本1",
            "index": 0,
            "finish_reason": "stop"
        },
        {
            "text": "生成的文本2",
            "index": 1,
            "finish_reason": "stop"
        }
    ]
}
```

## 示例

```python
import requests

url = "http://localhost:8000/v1/bert/mask_fill"  # 固定的API路径
data = {
    "model": "bert-base-chinese",
    "texts": ["巴黎是[MASK]国的首都", "北京是[MASK]国的首都"]
}

response = requests.post(url, json=data)
print(response.json())
```

### curl测试用例

你也可以使用curl命令行工具来测试服务：

```bash
curl -X POST "http://127.0.0.1:8000/v1/bart/completions"  \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bert-base-chinese",
    "texts": ["北京是[MASK]国的首都", "巴黎是[MASK]国的首都"],
    "max_tokens": 50,
    "temperature": 1.0
  }'
```

预期返回结果：

```json
{
    "id": "cmpl-20250410020017",
    "object": "text_completion",
    "created": 1744275617,
    "model": "bert-base-chinese",
    "choices": [
        {
            "text": "中",
            "index": 0,
            "finish_reason": "stop"
        },
        {
            "text": "法",
            "index": 1,
            "finish_reason": "stop"
        }
    ]
}
```

# classification 分类
**Endpoint:** `/v1/bert/classification`

**请求方法:** POST

**请求参数:**
```bash
curl -X POST http://localhost:8000/v1/bart/classification \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bert-base-chinese",
    "texts": ["这是一个很棒的电影", "这部电影太差劲了"],
    "labels": ["正面", "负面"],
    "max_tokens": 50,
    "temperature": 1.0
  }'
```

# ner 实体识别
**Endpoint:** `/v1/bert/ner`

**请求方法:** POST

```bash
curl -X POST "http://127.0.0.1:8000/v1/bart/ner" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bert-base-chinese",
    "texts": ["my name is roc", "比尔盖茨是微软的创始人"],
    "labels": ["PER", "ORG", "LOC"],
    "max_tokens": 50,
    "temperature": 1.0
  }'
```
# qa
**Endpoint:** `/v1/bert/qa`

**请求方法:** POST
```bash
curl -X POST "http://127.0.0.1:8000/v1/bart/qa" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bert-base-chinese",
    "questions": ["阿里巴巴的创始人是谁?"],
    "contexts": ["马云是阿里巴巴的创始人"],
    "max_tokens": 50,
    "temperature": 1.0
  }'
  ```