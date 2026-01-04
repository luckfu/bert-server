# BERT 推理服务

这是一个基于FastAPI的BERT模型推理服务，模拟了OpenAI API的接口格式。

## 安装

```bash
pip install -r requirements.txt
```

## 启动服务（基础用法）

`--model-path` 需要是本地模型目录（目录内包含 Transformers 的 config/tokenizer 等文件）。

服务启动时可以使用以下命令行参数（括号内为等价环境变量，便于 `uvicorn main:app` 部署）：

- `--service-name`（`SERVICE_NAME`）：服务名称，默认 `bert`；请求体 `model` 字段必须与之相同
- `--model-path`（`MODEL_PATH`）：模型目录路径（必需）
- `--task-type`（`TASK_TYPE`）：任务类型（必需），可选值：
  - `mask_fill`: 掩码填充任务
  - `classification`: 文本分类任务
  - `ner`: 命名实体识别任务
  - `qa`: 问答任务
- `--max-concurrency`（`MAX_CONCURRENCY`）：单进程最大并发（默认 5）
- `--device`（`DEVICE`）：推理设备，如 `cpu` / `cuda` / `cuda:0`
- `--dtype`（`DTYPE`）：混合精度 dtype：`auto|fp32|fp16|bf16`（默认 `auto`）
- `--tf32`（`TF32`）：CUDA TF32 开关：`1|0`（默认 1，仅 CUDA 生效）
- `--torch-compile`（`TORCH_COMPILE`）：torch.compile 开关：`1|0`（默认 0）
- `--torch-compile-mode`（`TORCH_COMPILE_MODE`）：torch.compile mode（默认 `reduce-overhead`）
- `--cpu-threads`（`CPU_THREADS`）：CPU intra-op 线程数（可选）
- `--cpu-interop-threads`（`CPU_INTEROP_THREADS`）：CPU inter-op 线程数（可选）
- `--host`：服务主机地址（默认 `0.0.0.0`）
- `--port`：服务端口号（默认 8000）

```bash
# 启动掩码填充服务（CPU）
python3 main.py --model-path ./bert-base-chinese --task-type mask_fill

# 启动文本分类服务（CPU）
python3 main.py --model-path ./bert-base-chinese --task-type classification

# 启动命名实体识别服务（CPU）
python3 main.py --model-path ./bert-base-chinese --task-type ner

# 启动问答服务（CPU）
python3 main.py --model-path ./bert-base-chinese --task-type qa
```

### CUDA 典型启动示例

```bash
# 使用 GPU、自动选择 bf16/fp16 混合精度、打开 TF32
python3 main.py \
  --model-path ./bert-base-chinese \
  --task-type classification \
  --device cuda:0 \
  --dtype auto \
  --tf32 1
```

### CPU 调优示例

```bash
# 固定 CPU 线程数，适用于多实例并发部署
python3 main.py \
  --model-path ./bert-base-chinese \
  --task-type ner \
  --device cpu \
  --cpu-threads 8 \
  --cpu-interop-threads 2
```

服务将在 http://localhost:8000 启动

## 部署（uvicorn）

如果你希望用 `uvicorn main:app` 方式启动，可以通过环境变量传参：

```bash
export MODEL_PATH=./bert-base-chinese
export TASK_TYPE=classification
export SERVICE_NAME=bert
export DEVICE=cuda:0
export DTYPE=auto
export TF32=1
export MAX_CONCURRENCY=5

uvicorn main:app --host 0.0.0.0 --port 8000
```

## 健康检查

**Endpoint:** `/health`

**请求方法:** GET

用于探活与查看当前 device / 混精 / compile 等运行参数。

## API 使用说明

注意：

- 每个进程只加载一种 `TASK_TYPE` 对应的模型；要同时提供多个任务，请启动多个进程/实例。
- `model` 字段必须等于 `--service-name`（默认 `bert`），否则会返回 400。
- `max_tokens` 在本服务里用作 tokenizer 的 `max_length`（截断长度），不是生成长度；`temperature` 当前未参与推理计算。

### 文本补全接口

**Endpoint:** `/v1/bert/mask_fill`

**请求方法:** POST

**请求参数:**

```json
{
    "model": "bert",  // BERT模型名称
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
    "model": "bert",
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
    "model": "bert",
    "texts": ["巴黎是[MASK]国的首都", "北京是[MASK]国的首都"]
}

response = requests.post(url, json=data)
print(response.json())
```

### curl测试用例

你也可以使用curl命令行工具来测试服务：

```bash
curl -X POST "http://127.0.0.1:8000/v1/bert/mask_fill"  \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bert",
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
    "model": "bert",
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

`labels` 可选；如传入，长度必须等于模型类别数，否则返回 400。

**请求参数:**
```bash
curl -X POST http://localhost:8000/v1/bert/classification \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bert",
    "texts": ["这是一个很棒的电影", "这部电影太差劲了"],
    "labels": ["正面", "负面"],
    "max_tokens": 50,
    "temperature": 1.0
  }'
```

# ner 实体识别
**Endpoint:** `/v1/bert/ner`

**请求方法:** POST

`labels` 可选；支持列表（按 0..N-1 映射）或字典（key 可为字符串，但必须可转成 int）。

```bash
curl -X POST "http://127.0.0.1:8000/v1/bert/ner" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bert",
    "texts": ["my name is roc", "比尔盖茨是微软的创始人"],
    "labels": ["PER", "ORG", "LOC"],
    "max_tokens": 50,
    "temperature": 1.0
  }'
```
# qa
**Endpoint:** `/v1/bert/qa`

**请求方法:** POST

`questions` 与 `contexts` 的长度必须一致。
```bash
curl -X POST "http://127.0.0.1:8000/v1/bert/qa" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bert",
    "questions": ["阿里巴巴的创始人是谁?"],
    "contexts": ["马云是阿里巴巴的创始人"],
    "max_tokens": 50,
    "temperature": 1.0
  }'
  ```
