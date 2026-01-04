import argparse
import asyncio
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from starlette.concurrency import run_in_threadpool
from transformers import (
    BertForMaskedLM,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertTokenizer,
)


class TaskType(Enum):
    MASK_FILL = "mask_fill"
    CLASSIFICATION = "classification"
    NER = "ner"
    QA = "qa"


@dataclass(frozen=True)
class ServiceConfig:
    service_name: str
    model_path: str
    task_type: TaskType
    max_concurrency: int
    device: torch.device
    autocast_device_type: Optional[Literal["cuda", "cpu"]]
    autocast_dtype: Optional[torch.dtype]
    tf32: bool
    torch_compile: bool
    torch_compile_mode: str
    cpu_threads: Optional[int]
    cpu_interop_threads: Optional[int]


def _parse_bool(raw: Optional[str], default: bool = False) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _parse_optional_int(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"无效整数: {raw}")


def _resolve_autocast(device: torch.device, dtype_raw: str) -> Tuple[Optional[Literal["cuda", "cpu"]], Optional[torch.dtype]]:
    dtype_raw = (dtype_raw or "auto").strip().lower()
    if dtype_raw in {"fp32", "float32", "none", "off"}:
        return None, None

    if device.type == "cuda":
        if dtype_raw in {"auto"}:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return "cuda", dtype
        if dtype_raw in {"fp16", "float16"}:
            return "cuda", torch.float16
        if dtype_raw in {"bf16", "bfloat16"}:
            return "cuda", torch.bfloat16
        raise RuntimeError(f"DTYPE 无效: {dtype_raw}")

    if device.type == "cpu":
        if dtype_raw in {"auto"}:
            return None, None
        if dtype_raw in {"bf16", "bfloat16"}:
            return "cpu", torch.bfloat16
        raise RuntimeError("CPU 仅支持 DTYPE=fp32/auto/bf16")

    return None, None


def _load_config_from_env() -> ServiceConfig:
    service_name = os.getenv("SERVICE_NAME", "bert")
    model_path = os.getenv("MODEL_PATH")
    task_type_raw = os.getenv("TASK_TYPE")
    max_concurrency_raw = os.getenv("MAX_CONCURRENCY", "5")
    device_raw = os.getenv("DEVICE")
    dtype_raw = os.getenv("DTYPE", "auto")
    tf32_raw = os.getenv("TF32", "1")
    torch_compile_raw = os.getenv("TORCH_COMPILE", "0")
    torch_compile_mode = os.getenv("TORCH_COMPILE_MODE", "reduce-overhead")
    cpu_threads_raw = os.getenv("CPU_THREADS")
    cpu_interop_threads_raw = os.getenv("CPU_INTEROP_THREADS")

    if not model_path:
        raise RuntimeError("MODEL_PATH 未设置")
    if not task_type_raw:
        raise RuntimeError("TASK_TYPE 未设置")

    try:
        task_type = TaskType(task_type_raw)
    except ValueError:
        raise RuntimeError(f"TASK_TYPE 无效: {task_type_raw}")

    try:
        max_concurrency = int(max_concurrency_raw)
    except ValueError:
        raise RuntimeError(f"MAX_CONCURRENCY 无效: {max_concurrency_raw}")
    if max_concurrency < 1:
        raise RuntimeError("MAX_CONCURRENCY 必须 >= 1")

    if device_raw:
        device = torch.device(device_raw)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autocast_device_type, autocast_dtype = _resolve_autocast(device, dtype_raw)
    tf32 = _parse_bool(tf32_raw, default=True)
    torch_compile = _parse_bool(torch_compile_raw, default=False)
    cpu_threads = _parse_optional_int(cpu_threads_raw)
    cpu_interop_threads = _parse_optional_int(cpu_interop_threads_raw)

    return ServiceConfig(
        service_name=service_name,
        model_path=model_path,
        task_type=task_type,
        max_concurrency=max_concurrency,
        device=device,
        autocast_device_type=autocast_device_type,
        autocast_dtype=autocast_dtype,
        tf32=tf32,
        torch_compile=torch_compile,
        torch_compile_mode=torch_compile_mode,
        cpu_threads=cpu_threads,
        cpu_interop_threads=cpu_interop_threads,
    )


def _load_tokenizer(model_path: str) -> BertTokenizer:
    return BertTokenizer.from_pretrained(model_path, use_fast=True)


def _load_model(task_type: TaskType, model_path: str) -> torch.nn.Module:
    if task_type == TaskType.MASK_FILL:
        return BertForMaskedLM.from_pretrained(model_path)
    if task_type == TaskType.CLASSIFICATION:
        return BertForSequenceClassification.from_pretrained(model_path)
    if task_type == TaskType.NER:
        return BertForTokenClassification.from_pretrained(model_path)
    if task_type == TaskType.QA:
        return BertForQuestionAnswering.from_pretrained(model_path)
    raise RuntimeError(f"不支持的任务类型: {task_type.value}")


app = FastAPI()


@app.on_event("startup")
def _startup() -> None:
    config = _load_config_from_env()
    if not os.path.exists(config.model_path):
        raise RuntimeError(f"模型路径不存在: {config.model_path}")

    if config.cpu_threads is not None:
        torch.set_num_threads(config.cpu_threads)
    if config.cpu_interop_threads is not None:
        torch.set_num_interop_threads(config.cpu_interop_threads)

    if config.device.type == "cuda" and config.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    tokenizer = _load_tokenizer(config.model_path)
    model = _load_model(config.task_type, config.model_path)
    model.to(config.device)
    model.eval()

    if config.torch_compile:
        try:
            model = torch.compile(model, mode=config.torch_compile_mode)
        except Exception:
            pass

    app.state.config = config
    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.semaphore = asyncio.Semaphore(config.max_concurrency)


def _get_runtime() -> Tuple[ServiceConfig, BertTokenizer, torch.nn.Module, asyncio.Semaphore]:
    try:
        return (
            app.state.config,
            app.state.tokenizer,
            app.state.model,
            app.state.semaphore,
        )
    except Exception:
        raise HTTPException(status_code=503, detail="服务未就绪：模型尚未加载")


def _validate_request_model(request_model: str, service_name: str) -> None:
    if request_model != service_name:
        raise HTTPException(status_code=400, detail=f"Model {request_model} not found")


def _require_task(actual: TaskType, required: TaskType) -> None:
    if actual != required:
        raise HTTPException(status_code=400, detail=f"当前服务只支持 {actual.value} 任务")

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
    labels: Optional[Union[Dict[int, str], List[str]]] = None

    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        if isinstance(v, str):
            return [v]
        return v
        
    @field_validator('labels')
    @classmethod
    def validate_labels(cls, v):
        if isinstance(v, list):
            return {i: label for i, label in enumerate(v)}
        if isinstance(v, dict):
            normalized: Dict[int, str] = {}
            for k, label in v.items():
                try:
                    normalized[int(k)] = label
                except Exception:
                    raise ValueError("labels 字典的 key 必须可转换为 int")
            return normalized
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

    @field_validator("contexts")
    @classmethod
    def validate_pairs(cls, contexts, info):
        questions = info.data.get("questions")
        if questions is not None and len(questions) != len(contexts):
            raise ValueError("questions 与 contexts 的长度必须一致")
        return contexts

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

async def _forward(**inputs):
    config, _, model, _ = _get_runtime()

    def _run():
        with torch.inference_mode():
            if config.autocast_device_type and config.autocast_dtype is not None:
                with torch.autocast(config.autocast_device_type, dtype=config.autocast_dtype):
                    return model(**inputs)
            return model(**inputs)

    outputs = await run_in_threadpool(_run)
    return outputs


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


async def create_mask_fill(request: CompletionRequest):
    config, tokenizer, _, semaphore = _get_runtime()
    async with semaphore:
        _validate_request_model(request.model, config.service_name)
        _require_task(config.task_type, TaskType.MASK_FILL)

        inputs = tokenizer(
            request.texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=request.max_tokens,
        )
        inputs = _to_device(inputs, config.device)

        outputs = await _forward(**inputs)

        input_ids = inputs["input_ids"]
        mask_token_id = tokenizer.mask_token_id
        decoded_outputs: List[str] = []

        for row_idx in range(input_ids.size(0)):
            mask_pos = (input_ids[row_idx] == mask_token_id).nonzero(as_tuple=True)[0]
            if mask_pos.numel() == 0:
                decoded_outputs.append("")
                continue
            logits = outputs.logits[row_idx][mask_pos]
            predicted_tokens = torch.argmax(logits, dim=-1)
            predicted_text = tokenizer.decode(predicted_tokens, skip_special_tokens=True).strip()
            decoded_outputs.append(predicted_text)

        choices = [
            {
                "text": output,
                "index": i,
                "finish_reason": "stop",
            }
            for i, output in enumerate(decoded_outputs)
        ]

        response = CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            object="text_completion",
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=choices,
        )

        return response


async def create_classification(request: ClassificationRequest):
    config, tokenizer, _, semaphore = _get_runtime()
    async with semaphore:
        _validate_request_model(request.model, config.service_name)
        _require_task(config.task_type, TaskType.CLASSIFICATION)

        inputs = tokenizer(
            request.texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=request.max_tokens,
        )
        inputs = _to_device(inputs, config.device)

        outputs = await _forward(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predicted_labels = torch.argmax(probs, dim=-1)

        label_count = probs.shape[1]
        labels = request.labels if request.labels else [str(i) for i in range(label_count)]
        if len(labels) != label_count:
            raise HTTPException(status_code=400, detail="labels 长度必须等于模型类别数")

        predictions_list = [
            {
                "label": labels[int(pred.item())],
                "score": float(prob[int(pred.item())]),
                "index": i,
            }
            for i, (pred, prob) in enumerate(zip(predicted_labels, probs))
        ]

        response = ClassificationResponse(
            id=f"cls-{uuid.uuid4().hex}",
            object="text_classification",
            created=int(datetime.now().timestamp()),
            model=request.model,
            predictions=predictions_list,
        )
        return response


def _extract_ner_entities(
    text: str,
    label_ids: torch.Tensor,
    probs: torch.Tensor,
    offsets: List[Tuple[int, int]],
    special_tokens_mask: List[int],
    label_map: Dict[int, str],
) -> List[dict]:
    entities: List[dict] = []
    current = None

    for token_idx, label_id_tensor in enumerate(label_ids):
        if special_tokens_mask[token_idx] == 1:
            continue
        start, end = offsets[token_idx]
        if start == end:
            continue

        label_id = int(label_id_tensor.item())
        if label_id == 0:
            if current is not None:
                current["text"] = text[current["start"] : current["end"]]
                current.pop("token_count", None)
                current.pop("label_id", None)
                entities.append(current)
                current = None
            continue

        label_name = label_map.get(label_id, str(label_id))
        confidence = float(probs[token_idx, label_id].item())

        if (
            current is not None
            and current["label_id"] == label_id
            and start == current["end"]
        ):
            token_count = int(current.get("token_count", 1))
            current["confidence"] = (current["confidence"] * token_count + confidence) / (token_count + 1)
            current["token_count"] = token_count + 1
            current["end"] = end
        else:
            if current is not None:
                current["text"] = text[current["start"] : current["end"]]
                current.pop("token_count", None)
                current.pop("label_id", None)
                entities.append(current)

            current = {
                "label_id": label_id,
                "label": label_name,
                "confidence": confidence,
                "start": start,
                "end": end,
                "token_count": 1,
            }

    if current is not None:
        current["text"] = text[current["start"] : current["end"]]
        current.pop("token_count", None)
        current.pop("label_id", None)
        entities.append(current)

    return entities


async def create_ner(request: NERRequest):
    config, tokenizer, _, semaphore = _get_runtime()
    async with semaphore:
        _validate_request_model(request.model, config.service_name)
        _require_task(config.task_type, TaskType.NER)

        label_map = request.labels or {}

        tokenized = tokenizer(
            request.texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=request.max_tokens,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )

        offsets_mapping = tokenized.pop("offset_mapping")
        special_tokens_mask = tokenized.pop("special_tokens_mask")
        inputs = _to_device(tokenized, config.device)

        outputs = await _forward(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_labels = torch.argmax(probs, dim=-1)

        entities_list = []
        for i, text in enumerate(request.texts):
            entities = _extract_ner_entities(
                text=text,
                label_ids=pred_labels[i].detach().cpu(),
                probs=probs[i].detach().cpu(),
                offsets=[(int(s), int(e)) for s, e in offsets_mapping[i].tolist()],
                special_tokens_mask=[int(x) for x in special_tokens_mask[i].tolist()],
                label_map=label_map,
            )
            entities_list.append(
                {
                    "text": text,
                    "entities": entities,
                    "index": i,
                }
            )

        response = NERResponse(
            id=f"ner-{uuid.uuid4().hex}",
            object="named_entity_recognition",
            created=int(datetime.now().timestamp()),
            model=request.model,
            entities=entities_list,
        )
        return response


def _best_span(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    max_answer_len: int = 30,
    top_k: int = 10,
) -> Tuple[int, int]:
    seq_len = start_logits.size(0)
    k = min(top_k, seq_len)
    start_top = torch.topk(start_logits, k=k).indices.tolist()
    end_top = torch.topk(end_logits, k=k).indices.tolist()

    best_s, best_e = 0, 0
    best_score = float("-inf")
    for s in start_top:
        for e in end_top:
            if e < s:
                continue
            if e - s + 1 > max_answer_len:
                continue
            score = float(start_logits[s].item() + end_logits[e].item())
            if score > best_score:
                best_score = score
                best_s, best_e = s, e
    return best_s, best_e


async def create_qa(request: QARequest):
    config, tokenizer, _, semaphore = _get_runtime()
    async with semaphore:
        _validate_request_model(request.model, config.service_name)
        _require_task(config.task_type, TaskType.QA)

        inputs = tokenizer(
            request.questions,
            request.contexts,
            return_tensors="pt",
            max_length=request.max_tokens,
            truncation=True,
            padding=True,
        )
        inputs = _to_device(inputs, config.device)

        outputs = await _forward(**inputs)
        answers_list = []
        for i, (question, context) in enumerate(zip(request.questions, request.contexts)):
            s, e = _best_span(outputs.start_logits[i].detach().cpu(), outputs.end_logits[i].detach().cpu())
            answer_tokens = inputs["input_ids"][i].detach().cpu()[s : e + 1]
            answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            answers_list.append(
                {
                    "question": question,
                    "context": context,
                    "answer": answer_text,
                    "index": i,
                }
            )

        response = QAResponse(
            id=f"qa-{uuid.uuid4().hex}",
            object="question_answering",
            created=int(datetime.now().timestamp()),
            model=request.model,
            answers=answers_list,
        )
        return response


@app.get("/health")
def health():
    config, _, _, _ = _get_runtime()
    return {
        "service_name": config.service_name,
        "task_type": config.task_type.value,
        "device": str(config.device),
        "autocast_dtype": str(config.autocast_dtype) if config.autocast_dtype is not None else None,
        "tf32": config.tf32 if config.device.type == "cuda" else None,
        "torch_compile": config.torch_compile,
    }


def _register_routes() -> None:
    prefix = "/v1/bert"
    app.post(f"{prefix}/mask_fill", response_model=CompletionResponse)(create_mask_fill)
    app.post(f"{prefix}/classification", response_model=ClassificationResponse)(create_classification)
    app.post(f"{prefix}/ner", response_model=NERResponse)(create_ner)
    app.post(f"{prefix}/qa", response_model=QAResponse)(create_qa)


_register_routes()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT 推理服务")
    parser.add_argument("--service-name", type=str, default="bert", help="服务名称")
    parser.add_argument("--model-path", type=str, required=True, help="模型目录路径")
    parser.add_argument("--task-type", type=str, required=True, choices=[t.value for t in TaskType], help="任务类型")
    parser.add_argument("--max-concurrency", type=int, default=5, help="最大并发请求数")
    parser.add_argument("--device", type=str, default=None, help="推理设备，如 cpu/cuda/cuda:0")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="自动混精推理 dtype")
    parser.add_argument("--tf32", type=int, default=1, choices=[0, 1], help="CUDA TF32 开关(1/0)")
    parser.add_argument("--torch-compile", type=int, default=0, choices=[0, 1], help="torch.compile 开关(1/0)")
    parser.add_argument("--torch-compile-mode", type=str, default="reduce-overhead", help="torch.compile mode")
    parser.add_argument("--cpu-threads", type=int, default=None, help="CPU intra-op 线程数")
    parser.add_argument("--cpu-interop-threads", type=int, default=None, help="CPU inter-op 线程数")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口号")

    args = parser.parse_args()
    os.environ["SERVICE_NAME"] = args.service_name
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["TASK_TYPE"] = args.task_type
    os.environ["MAX_CONCURRENCY"] = str(args.max_concurrency)
    os.environ["DTYPE"] = args.dtype
    os.environ["TF32"] = str(args.tf32)
    os.environ["TORCH_COMPILE"] = str(args.torch_compile)
    os.environ["TORCH_COMPILE_MODE"] = args.torch_compile_mode
    if args.cpu_threads is not None:
        os.environ["CPU_THREADS"] = str(args.cpu_threads)
    if args.cpu_interop_threads is not None:
        os.environ["CPU_INTEROP_THREADS"] = str(args.cpu_interop_threads)
    if args.device:
        os.environ["DEVICE"] = args.device

    import uvicorn

    print(
        f"服务启动成功！访问地址: http://{args.host}:{args.port}/v1/bert/{args.task_type}，当前使用模型: {args.service_name}"
    )
    uvicorn.run(app, host=args.host, port=args.port)
