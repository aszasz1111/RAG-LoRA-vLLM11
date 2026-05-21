import os
import sys
import json
import csv
import time
from datetime import datetime

# 让 eval/rag_eval.py 可以导入上一级目录的 app.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

import app


# ===================== 配置区 =====================
QUESTION_FILE = os.path.join(CURRENT_DIR, "eval_questions.json")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "results")
# =================================================


def load_questions():
    """读取评测问题集"""
    with open(QUESTION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def keyword_hit_score(answer, expected_keywords):
    """
    简单关键词命中率：
    命中的关键词数 / 总关键词数
    若 expected_keywords 为空，则返回 None
    """
    if not expected_keywords:
        return None, []

    hit_keywords = []
    for keyword in expected_keywords:
        if keyword in answer:
            hit_keywords.append(keyword)

    score = len(hit_keywords) / len(expected_keywords)
    return round(score, 3), hit_keywords


def generate_answer(question):
    """
    复用 app.py 中的 retriever 与 llm，
    对单个问题进行 RAG 问答。
    """

    # 1. 检索相关文档
    relevant_docs = app.retriever.invoke(question)

    # 2. 若没有召回
    if not relevant_docs:
        answer = "抱歉，我在当前文档中没有检索到与该问题足够相关的内容。"
        sources = "未检索到相关文档"
        retrieved_count = 0
        return answer, sources, retrieved_count

    # 3. 构造上下文
    context = app.build_context(relevant_docs)
    source_text = app.get_source_text(relevant_docs)
    retrieved_count = len(relevant_docs)

    # 4. 构造评测使用的 Prompt
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

    # 5. 调用模型
    res = app.llm.invoke(prompt)
    answer = res.content.strip()

    return answer, source_text, retrieved_count


def save_to_csv(results, csv_path):
    """保存为 CSV，方便 Excel 打开分析"""
    fieldnames = [
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

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in results:
            writer.writerow({
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


def save_to_json(results, json_path):
    """保存为 JSON，方便后续做可视化或进一步分析"""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def print_summary(results):
    """打印评测摘要"""
    total = len(results)
    no_retrieval_count = sum(1 for item in results if item["retrieved_count"] == 0)

    scores = [
        item["keyword_hit_score"]
        for item in results
        if item["keyword_hit_score"] is not None
    ]

    avg_score = round(sum(scores) / len(scores), 3) if scores else 0

    avg_time = round(
        sum(item["response_time_seconds"] for item in results) / total,
        3
    ) if total else 0

    print("\n================= 评测结果摘要 =================")
    print(f"评测问题总数：{total}")
    print(f"未召回文档的问题数：{no_retrieval_count}")
    print(f"关键词平均命中率：{avg_score}")
    print(f"平均响应耗时：{avg_time} 秒")
    print("===============================================")


def main():
    print("正在初始化知识库与模型...")

    if not app.init_knowledge_base():
        print("初始化失败，评测终止。")
        return

    questions = load_questions()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_output = os.path.join(OUTPUT_DIR, f"rag_eval_{timestamp}.csv")
    json_output = os.path.join(OUTPUT_DIR, f"rag_eval_{timestamp}.json")

    results = []

    print(f"\n开始评测，共 {len(questions)} 个问题...\n")

    for idx, item in enumerate(questions, start=1):
        qid = item["id"]
        category = item["category"]
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])

        print(f"[{idx}/{len(questions)}] 正在评测：{question}")

        start_time = time.time()
        answer, sources, retrieved_count = generate_answer(question)
        end_time = time.time()

        response_time = round(end_time - start_time, 3)

        score, hit_keywords = keyword_hit_score(answer, expected_keywords)

        result = {
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

        results.append(result)

        print(f"    召回文本块数：{retrieved_count}")
        print(f"    关键词命中率：{score}")
        print(f"    耗时：{response_time} 秒\n")

    save_to_csv(results, csv_output)
    save_to_json(results, json_output)

    print_summary(results)

    print("\n评测结果已保存：")
    print(f"CSV：{csv_output}")
    print(f"JSON：{json_output}")


if __name__ == "__main__":
    main()