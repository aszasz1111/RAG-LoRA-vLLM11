# 🚀 RAG + LoRA + vLLM End-to-End LLM System

## 📌 项目简介

本项目构建了一个**端到端大模型应用系统**，融合了：

- 📚 RAG（Retrieval-Augmented Generation）知识增强
- 🧠 LoRA（Low-Rank Adaptation）轻量化微调
- ⚡ vLLM 高性能推理加速

实现了“**知识库检索 + 微调增强 + 高效推理输出**”的一体化LLM应用系统，可用于问答系统、知识增强生成等场景。

---

## 🧠 系统架构
```markdown
整体流程如下：
用户输入
↓
Query Embedding
↓
ChromaDB 向量检索（RAG）
↓
上下文拼接 Prompt
↓
LoRA 微调模型增强生成能力
↓
vLLM 高速推理输出结果
```
---

## ⚙️ 技术栈

- Python
- PyTorch
- HuggingFace Transformers
- PEFT（LoRA）
- ChromaDB（向量数据库）
- vLLM（推理加速框架）
- OpenAI / LLM Base Model

---


---

## 二、增强版：带详细注释的结构（适合公开仓库，更清晰）
如果想让别人一眼看懂每个模块的作用，可以加上层级说明：
```markdown
# 项目结构
```text
RAG+LoRA+vLLM/
├── app.py                  # 🌐 Web/API服务入口（Gradio/Flask部署）
├── main.py                 # 🚀 主推理入口，整合RAG+微调模型+LLM
├── requirements.txt        # 📦 项目依赖清单，一键安装所有依赖
├── .gitignore              # 🚫 Git忽略配置，过滤大文件/缓存/本地数据
├── README.md               # 📖 项目说明文档
│
├── finetune/               # ⚙️ LoRA微调相关代码
│   ├── train.py            # 微调主脚本，支持QLoRA低秩适配
│   └── configs/           # 微调配置文件（学习率、LoRA参数等）
│
├── eval/                   # 📊 模型评估模块
│   ├── metrics.py          # 评估指标（准确率、召回率、F1值）
│   └── results/           # 评估结果输出目录，自动生成报告
│
├── docs/                   # 📚 项目文档与数据
│   ├── raw_data/           # 训练/测试原始数据集
│   └── usage_notes.md      # 项目使用说明、环境搭建步骤
│
├── chroma_db/              # 🗄️ 本地向量数据库（运行RAG时自动生成，无需上传）
└── finetune.zip            # 📁 训练备份文件（已忽略，本地保留即可）
```

---

## 🧪 核心功能

### 1️⃣ RAG 知识增强
- 文档切分（chunking）
- embedding 向量化
- ChromaDB 检索 top-k 相关内容
- 增强 LLM 输入上下文

---

### 2️⃣ LoRA 微调
- 基于预训练大模型进行轻量微调
- 仅训练低秩矩阵（节省显存）
- 提升领域任务表现

---

### 3️⃣ vLLM 推理加速
- PagedAttention 优化显存管理
- 支持高并发推理
- 提升生成速度与吞吐

---

## 📊 评估模块（Eval）

支持对模型进行：
- Accuracy（准确率）
- Recall（召回率）
- Response quality evaluation

输出结果保存在：
eval/results/*.csv

---

## 🚀 快速启动

### 1️⃣ 安装依赖
```bash
pip install -r requirements.txt
```
### 2️⃣ 构建向量数据库（RAG）
python finetune/build_db.py
### 3️⃣ 启动推理服务
python app.py

---

## 📌 项目亮点
🔥 端到端 LLM 应用系统（非单一模型实验）
🔥 RAG + LoRA + vLLM 三层增强架构
🔥 支持知识增强 + 领域微调 + 高速推理
🔥 工程化完整（可部署/可扩展）

## 💡 可优化方向（面试加分点）
引入 rerank 模型提升 RAG 精度
使用 FAISS 替代/对比 ChromaDB
加入 Prompt Template 管理系统
Docker 部署
FastAPI + 前端可视化界面
