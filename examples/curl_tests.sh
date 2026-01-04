#!/bin/bash

# Mask Fill测试用例
echo "Testing mask_fill endpoint..."
curl -X POST "http://127.0.0.1:8000/v1/bert/mask_fill" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bert",
    "texts": ["北京是[MASK]国的首都", "巴黎是[MASK]国的首都"],
    "max_tokens": 50,
    "temperature": 1.0
  }'

echo "\n\nTesting classification endpoint..."
curl -X POST "http://127.0.0.1:8000/v1/bert/classification" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bert",
    "texts": ["这个产品非常好", "服务很差"],
    "labels": ["正面", "负面"],
    "max_tokens": 50,
    "temperature": 1.0
  }'

echo "\n\nTesting NER endpoint..."
curl -X POST "http://127.0.0.1:8000/v1/bert/ner" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bert",
    "texts": ["马云是阿里巴巴的创始人", "比尔盖茨是微软的创始人"],
    "labels": ["PER", "ORG", "LOC"],
    "max_tokens": 50,
    "temperature": 1.0
  }'

echo "\n\nTesting QA endpoint..."
curl -X POST "http://127.0.0.1:8000/v1/bert/qa" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bert",
    "questions": ["阿里巴巴的创始人是谁?"],
    "contexts": ["马云是阿里巴巴的创始人"],
    "max_tokens": 50,
    "temperature": 1.0
  }'
