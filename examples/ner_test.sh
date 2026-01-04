#!/bin/bash

# 测试命名实体识别接口的curl命令
curl -X POST http://localhost:8000/v1/bert/ner \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bert",
    "texts": ["李明在北京大学计算机系读研究生", "华为公司的任正非今天在深圳总部召开会议"],
    "max_tokens": 50,
    "temperature": 1.0
  }'
