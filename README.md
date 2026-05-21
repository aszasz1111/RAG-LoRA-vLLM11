# Meteorology RAG + LoRA + vLLM System

## Project Overview

基于气象领域论文构建的智能问答系统。

技术栈：
- LangChain
- Chroma
- SentenceTransformer
- Qwen3-4B
- QLoRA
- vLLM
- Gradio

## Features

- PDF知识库构建
- RAG检索增强
- 气象领域LoRA微调
- vLLM推理部署
- 自动化评测

## Results

RAG评测：

- Recall: 100%
- Keyword Hit Rate: 85.4%

LoRA训练：

- Train Loss: 1.528
- Eval Loss: 1.184

vLLM推理：

- 平均响应时间：10.09s
- 3路并发总耗时：10.57s