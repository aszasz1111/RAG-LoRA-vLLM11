# 🚀 RAG+LoRA+vLLM 气象领域智能问答系统

## 📌 项目简介

本项目构建了一个**端到端大模型应用系统**，融合了：

- 📚 RAG（Retrieval-Augmented Generation）知识增强
- 🧠 LoRA（Low-Rank Adaptation）轻量化微调
- ⚡ vLLM 高性能推理加速

实现了“**知识库检索 + 微调增强 + 高效推理输出**”的一体化LLM应用系统，可用于问答系统、知识增强生成等场景。

本项目以火山方舟 豆包轻量化开源大模型为基座，
利用ChromaDB构建气象专业知识库向量库，实现文档语义检索；
通过QLoRA低显存微调，适配气象雷达、短时预报、雷暴大风等垂直领域场景；
依托vLLM引擎实现高并发、低延迟流式推理，兼容 OpenAI 标准接口；
内置完整评估脚本，支持分块策略对比、问答效果评测、并发压力测试。

---
## ✨ 项目亮点
- 优质中文基座：选用火山开源豆包模型，中文语义理解、书面文献解析能力优异，天然适配中文气象专业资料
- 低门槛领域适配：采用 QLoRA 量化微调方案，无需高端算力，普通消费级显卡即可完成行业领域模型微调
- 彻底缓解模型幻觉：结合 RAG 检索增强，依托本地专业知识库作答，输出内容有据可依，贴合气象学术研究内容
- 极速高并发推理：基于 vLLM PagedAttention 加速机制，推理响应快、吞吐量大，支持多用户同时请求
- 全流程可量化评测：从知识库构建参数、召回效果到模型生成速度均支持量化对比，方便调优迭代
- 多场景一键部署：同时支持本地离线运行、Colab 云端部署，上手简单，开箱即用
  
---

## 🎯 项目特色
### 垂直领域高度定制
深耕气象行业场景，聚焦双偏振雷达参数、强对流天气、短时预报、微下击暴流等专业研究方向，定制专属抽取问答任务。
### 模块化解耦架构
知识库检索、模型微调、高速推理、效果评估四大模块完全独立，代码结构清晰，便于二次开发与功能拓展。
### 多元交互使用方式
支持终端命令行调用、Gradio 可视化网页对话、后端 API 接口调用三种使用形式，满足调试、演示、项目集成不同需求。
### 完善性能测试体系
内置单轮耗时测试、多轮串行测试、多线程并发压测，精准统计模型推理效率，适配线上服务落地需求。
### 轻量化落地友好
选用小参数量开源豆包模型搭配 LoRA 微调，整体资源占用低，部署成本低，适合学生学习、科研实验与小型业务落地。
### 标准化任务输出
固定气象文献信息抽取格式，可自动提炼研究主题、研究对象、数据资料、研究方法、核心结论，实用性极强。

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
## 项目整体流程
用户提问 → 问题向量化 → 气象知识库语义检索 → 拼接上下文 Prompt → 豆包领域 LoRA 模型增强理解 → vLLM 高速生成回答 → 输出专业结构化答案

## ⚙️ 技术栈

- 基础框架：Python 3.9+
- 大模型生态：Transformers、PEFT、Bitsandbytes
- 微调方案：QLoRA 轻量化微调
- 检索引擎：LangChain + ChromaDB 向量数据库
- 高速推理：vLLM（PagedAttention 加速）
- 接口规范：兼容 OpenAI Chat 接口
- 评估工具：自定义 RAG 问答评测、耗时测速、并发测试

---

##  项目结构
```markdown
RAG+LoRA+vLLM/
├── app.py                  # 🌐 Web/API服务入口（Gradio/Flask部署）
├── main.py                 # 🚀 主推理入口，整合RAG+微调模型+LLM
├── requirements.txt        # 📦 项目依赖清单，一键安装所有依赖
├── .gitignore              # 🚫 Git忽略配置，过滤大文件/缓存/本地数据
├── README.md               # 📖 项目说明文档
│
├── finetune/                       # LoRA微调模块
│   ├── data/                       # 微调训练/验证数据集
│   │   ├── weather_sft_train.json
│   │   └── weather_sft_eval.json
│   ├── dataset_info.json           # 数据集配置
│   └── qwen_lora_sft.yaml          # 微调超参配置文件
│
├── eval/                           # RAG效果评估模块
│   ├── rag_eval.py                 # 问答结果量化评估
│   ├── rag_param_experiment.py     # 分块/检索参数对照实验
│   ├── eval_questions.json         # 评测问答数据集
│   └── temp_vector_dbs/            # 不同参数实验向量库
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

### 气象文档知识库构建
自动解析 PDF/Word 气象资料，文本分块、向量化入库，支持自定义分块大小、重叠长度、检索条数。
### 领域轻量化微调
使用 QLoRA4/8 比特量化训练，低显存即可完成气象专业话术、文献信息抽取能力适配。
### vLLM 高并发推理服务
部署开源大模型 + 加载私有 LoRA 权重，支持批量请求、流式输出，大幅提升问答响应速度。
### 标准 OpenAI 接口调用
无需改动业务代码，直接使用 openai 库调用本地 vLLM 服务，无缝对接各类上层应用。
### 完整效果评测体系
支持 RAG 召回效果评测、模型回答准确率打分、单轮耗时测试、多线程并发压力测试。


---

## 🚀 快速启动-环境部署

### 安装依赖
```bash
# 创建虚拟环境
python -m venv .venv
# 激活环境
.venv\Scripts\activate
# 安装依赖
pip install -r requirements.txt
```
### Colab 云端环境前置配置
```bash
# 挂载谷歌云盘(存放LoRA微调权重)
from google.colab import drive
drive.mount('/content/drive')

# 设置HuggingFace国内镜像加速下载
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 升级vLLM至最新稳定版
!pip install -U vllm openai
```
### vLLM+LoRA 服务一键启动命令
```bash
!pkill -f "vllm serve"
!rm -f /content/vllm_server.log

!nohup vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --dtype half \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.70 \
  --port 8000 \
  --enable-lora \
  --max-lora-rank 8 \
  --lora-modules weather_lora=/content/drive/MyDrive/RAG_LoRA_vLLM/qwen3-4b-weather-qlora-sft \
  > /content/vllm_server.log 2>&1 &

# 等待模型加载完成
!sleep 30
# 查看启动日志
!tail -n 100 /content/vllm_server.log
```
### 构建向量数据库（RAG）
python finetune/build_db.py
### 启动推理服务
python app.py

---
## 接口调用示例
### 服务连通性检测
```bash
import requests
r = requests.get("http://127.0.0.1:8000/v1/models")
print(r.status_code, r.json())
```
### 专业文献信息抽取调用
```bash
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1"
)

res = client.chat.completions.create(
    model="weather_lora",
    messages=[
        {"role":"user","content":"请提取研究主题、研究对象、数据资料、研究方法、核心结论：某研究利用双偏振天气雷达资料，对弱垂直风切变背景下的局地微下击暴流过程进行分析，重点研究KDP核、ZDR柱和ZDR槽的演变特征。结果表明，融化层附近KDP核增强和ZDR柱减弱下降，对下沉气流爆发具有一定指示意义。"}
    ],
    temperature=0.1,
    max_tokens=512
)
print(res.choices[0].message.content)
```


