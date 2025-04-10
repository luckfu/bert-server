#!/bin/bash

# 测试文本分类接口的curl命令
curl -X POST http://localhost:8000/v1/bart/classification \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bert-base-chinese",
    "texts": ["这是一个很棒的电影", "这部电影太差劲了"],
    "labels": ["正面", "负面"],
    "max_tokens": 50,
    "temperature": 1.0
  }'