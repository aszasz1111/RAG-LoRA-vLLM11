import os
import sys
import json
import csv
import time
import shutil
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 让 eval/rag_param_experiment.py 可以导入上一级目录 app.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

import app


# ===================== 路径配置 =====================
QUESTION_FILE = os.path.join(CURRENT_DIR, "eval_questions.json")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "param_experiment_results")
TEMP_DB_ROOT = os.path.join(CURRENT_DIR, "temp_vector_dbs")

DOCS_FOLDER = app.DOCS_FOLDER
# ====================================================


# ===================== 参数实验组 =====================
EXPERIMENT_CONFIGS = [
    {
        "name": "A_baseline_chunk800_overlap100_top5",
        "chunk_size": 800,
        "chunk_overlap": 100,
        "top_k": 5
    },
    {
        "name": "B_chunk500_overlap80_top5",
        "chunk_size": 500,
        "chunk_overlap": 80,
        "top_k": 5
    },
    {
        "name": "C_chunk1000_overlap150_top5",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5
    },
    {
        "name": "D_chunk800_overlap100_top3",
        "chunk_size": 800,
        "chunk_overlap": 100,
        "top_k": 3
    }
]
# ====================================================


def load_questions():
    """读取评测问题"""
    with open(QUESTION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def count_pdf_files(folder):
    """统计 PDF 文件数量"""
    return sum(
        1
        for root, _, files in os.walk(folder)
        for file in files
        if file.lower().endswith(".pdf")
    )


def keyword_hit_score(answer, expected_keywords):
    """
    简单关键词命中率：
    命中关键词数量 / 预期关键词总数
    """
    if not expected_keywords:
        return None, []

    hit_keywords = []

    for keyword in expected_keywords:
        if keyword in answer:
            hit_keywords.append(keyword)

    score = len(hit_keywords) / len(expected_keywords)
    return round(score, 3), hit_keywords


def build_vector_db_for_config(config):
    """
    根据某一组参数重新构建临时向量库
    """
    experiment_name = config["name"]
    chunk_size = config["chunk_size"]
    chunk_overlap = config["chunk_overlap"]

    db_path = os.path.join(TEMP_DB_ROOT, experiment_name)

    # 若该组临时向量库已存在，先删除，保证每次实验重新构建
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    print(f"\n{'=' * 70}")
    print(f"正在构建向量库：{experiment_name}")
    print(f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print(f"{'=' * 70}")

    pdf_count = count_pdf_files(DOCS_FOLDER)
    print(f"PDF 文件数：{pdf_count}")

    loader = DirectoryLoader(
        DOCS_FOLDER,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        loader_kwargs={"mode": "page"},
        show_progress=True
    )

    docs = loader.load()

    if not docs:
        raise RuntimeError("PDF 文档加载失败，无法继续实验。")

    print(f"按页加载后的文档单元数：{len(docs)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""]
    )

    splits = text_splitter.split_documents(docs)

    print(f"切分后的文本块数：{len(splits)}")

    if not splits:
        raise RuntimeError("文本切分结果为空，无法建立向量库。")

    print("正在生成临时向量库...")

    db = Chroma.from_documents(
        documents=splits,
        embedding=app.embeddings,
        persist_directory=db_path
    )

    print("临时向量库构建完成。")

    return db, len(splits)


def generate_answer(question, retriever):
    """
    基于指定 retriever 生成答案
    """
    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        answer = "抱歉，我在当前文档中没有检索到与该问题足够相关的内容。"
        sources = "未检索到相关文档"
        retrieved_count = 0
        return answer, sources, retrieved_count

    context = app.build_context(relevant_docs)
    source_text = app.get_source_text(relevant_docs)

    prompt = f"""
你是一个面向气象论文、雷达资料和专业报告的 PDF 智能问答助手。

请严格遵守以下规则：
1. 只能依据“参考资料”中的内容回答。
2. 不允许凭空补充文档中没有的信息。
3. 若参考资料不足以回答，直接说明：“抱歉，我在文档中没有找到足够明确的相关内容。”
4. 回答应尽量清晰、准确。
5. 不要伪造引用。
6. 若问题涉及多个资料，可综合归纳。

【参考资料】
{context}

【用户问题】
{question}

请给出回答：
"""

    res = app.llm.invoke(prompt)
    answer = res.content.strip()

    return answer, source_text, len(relevant_docs)


def run_single_experiment(config, questions):
    """
    执行单组参数实验
    """
    db, split_count = build_vector_db_for_config(config)

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config["top_k"]}
    )

    experiment_results = []

    print(f"\n开始评测实验组：{config['name']}")
    print(f"共 {len(questions)} 个问题。\n")

    for idx, item in enumerate(questions, start=1):
        qid = item["id"]
        category = item["category"]
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])

        print(f"[{idx}/{len(questions)}] {question}")

        start_time = time.time()
        answer, sources, retrieved_count = generate_answer(question, retriever)
        end_time = time.time()

        response_time = round(end_time - start_time, 3)
        score, hit_keywords = keyword_hit_score(answer, expected_keywords)

        result = {
            "experiment_name": config["name"],
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "top_k": config["top_k"],
            "split_count": split_count,

            "id": qid,
            "category": category,
            "question": question,
            "answer": answer,
            "expected_keywords": expected_keywords,
            "hit_keywords": hit_keywords,
            "keyword_hit_score": score,
            "retrieved_count": retrieved_count,
            "sources": sources,
            "response_time_seconds": response_time
        }

        experiment_results.append(result)

        print(f"    召回文本块数：{retrieved_count}")
        print(f"    关键词命中率：{score}")
        print(f"    耗时：{response_time} 秒\n")

    return experiment_results


def summarize_experiment(config, results):
    """
    汇总单组实验结果
    """
    total = len(results)

    no_retrieval_count = sum(
        1 for item in results
        if item["retrieved_count"] == 0
    )

    valid_scores = [
        item["keyword_hit_score"]
        for item in results
        if item["keyword_hit_score"] is not None
    ]

    avg_keyword_score = (
        round(sum(valid_scores) / len(valid_scores), 3)
        if valid_scores else 0
    )

    avg_response_time = (
        round(
            sum(item["response_time_seconds"] for item in results) / total,
            3
        )
        if total else 0
    )

    split_count = results[0]["split_count"] if results else 0

    return {
        "experiment_name": config["name"],
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
        "top_k": config["top_k"],
        "split_count": split_count,
        "question_count": total,
        "no_retrieval_count": no_retrieval_count,
        "avg_keyword_hit_score": avg_keyword_score,
        "avg_response_time_seconds": avg_response_time
    }


def save_detail_results(all_results, output_path):
    """保存所有实验明细 CSV"""
    fieldnames = [
        "experiment_name",
        "chunk_size",
        "chunk_overlap",
        "top_k",
        "split_count",
        "id",
        "category",
        "question",
        "answer",
        "expected_keywords",
        "hit_keywords",
        "keyword_hit_score",
        "retrieved_count",
        "sources",
        "response_time_seconds"
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in all_results:
            writer.writerow({
                "experiment_name": item["experiment_name"],
                "chunk_size": item["chunk_size"],
                "chunk_overlap": item["chunk_overlap"],
                "top_k": item["top_k"],
                "split_count": item["split_count"],
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "answer": item["answer"],
                "expected_keywords": "、".join(item["expected_keywords"]),
                "hit_keywords": "、".join(item["hit_keywords"]),
                "keyword_hit_score": item["keyword_hit_score"],
                "retrieved_count": item["retrieved_count"],
                "sources": item["sources"],
                "response_time_seconds": item["response_time_seconds"]
            })


def save_summary_results(summary_results, output_path):
    """保存实验汇总 CSV"""
    fieldnames = [
        "experiment_name",
        "chunk_size",
        "chunk_overlap",
        "top_k",
        "split_count",
        "question_count",
        "no_retrieval_count",
        "avg_keyword_hit_score",
        "avg_response_time_seconds"
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in summary_results:
            writer.writerow(item)


def save_json(data, output_path):
    """保存 JSON"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_final_summary(summary_results):
    """终端打印总结果"""
    print("\n")
    print("=" * 90)
    print("RAG 参数对比实验汇总")
    print("=" * 90)

    for item in summary_results:
        print(
            f"{item['experiment_name']} | "
            f"chunk={item['chunk_size']} | "
            f"overlap={item['chunk_overlap']} | "
            f"top_k={item['top_k']} | "
            f"文本块={item['split_count']} | "
            f"平均命中率={item['avg_keyword_hit_score']} | "
            f"平均耗时={item['avg_response_time_seconds']}s"
        )

    print("=" * 90)


def main():
    print("正在初始化大模型...")
    print("本实验会自行重建临时向量库，不会改动主程序 chroma_db。")

    # 这里只需要初始化 llm，不依赖 app.retriever
    # 复用 app.init_knowledge_base()，确保 llm 与 embeddings 都可用
    if not app.init_knowledge_base():
        print("初始化失败，实验终止。")
        return

    questions = load_questions()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DB_ROOT, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    detail_csv = os.path.join(
        OUTPUT_DIR,
        f"rag_param_experiment_detail_{timestamp}.csv"
    )

    summary_csv = os.path.join(
        OUTPUT_DIR,
        f"rag_param_experiment_summary_{timestamp}.csv"
    )

    detail_json = os.path.join(
        OUTPUT_DIR,
        f"rag_param_experiment_detail_{timestamp}.json"
    )

    summary_json = os.path.join(
        OUTPUT_DIR,
        f"rag_param_experiment_summary_{timestamp}.json"
    )

    all_results = []
    summary_results = []

    for config in EXPERIMENT_CONFIGS:
        results = run_single_experiment(config, questions)
        summary = summarize_experiment(config, results)

        all_results.extend(results)
        summary_results.append(summary)

    save_detail_results(all_results, detail_csv)
    save_summary_results(summary_results, summary_csv)

    save_json(all_results, detail_json)
    save_json(summary_results, summary_json)

    print_final_summary(summary_results)

    print("\n实验结果已保存：")
    print(f"明细 CSV：{detail_csv}")
    print(f"汇总 CSV：{summary_csv}")
    print(f"明细 JSON：{detail_json}")
    print(f"汇总 JSON：{summary_json}")


if __name__ == "__main__":
    main()