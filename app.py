from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

import os
import shutil
import warnings
import gradio as gr

warnings.filterwarnings("ignore")

# ===================== 配置区（主要改这里）=====================

# 1. 火山方舟 API Key
API_KEY = "ark-afe3fa9a-1912-4543-9409-a76402cbb69b-6f1d8"

# 2. 火山方舟豆包 LLM 接入点 ID
# 你当前这个值保持即可
LLM_MODEL_ID = "ep-20260518213229-p96l9"

# 3. 项目路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_FOLDER = os.path.join(BASE_DIR, "docs")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# 4. 是否重建向量库
# 注意：你这次从云端 Embedding 改成本地 Embedding，
# 第一次运行必须设置为 True。
# 成功建库后，再改回 False。
REBUILD_VECTOR_DB = False

# 5. 文本切分参数
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# 6. 检索参数
TOP_K = 5

# 7. 最多保留最近几条历史消息参与问答
MAX_HISTORY_MESSAGES = 6

# ===============================================================


# ===================== 本地免费嵌入模型 =====================
class LocalEmbeddings:
    """
    使用本地 SentenceTransformer 生成向量。
    不调用任何 Embedding API，不产生额外接口费用。
    """

    def __init__(self):
        print("📦 正在加载本地嵌入模型 BAAI/bge-base-zh-v1.5 ...")
        self.model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
        print("✅ 本地嵌入模型加载完成")

    def embed_documents(self, texts):
        if not texts:
            return []

        texts = [text.strip() for text in texts if text and text.strip()]

        if not texts:
            return []

        print(f"🧠 正在进行本地向量化，共 {len(texts)} 个文本块...")

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True
        ).tolist()

        print(f"✅ 本地向量化完成，共生成 {len(embeddings)} 条向量")
        return embeddings

    def embed_query(self, text):
        if not text or not text.strip():
            return []

        embeddings = self.embed_documents([text.strip()])
        return embeddings[0]


# ===================== 来源处理 =====================
def get_source_text(docs):
    """
    输出引用来源：
    1. xxx.pdf（第1页）
    2. yyy.pdf（第3页）
    """
    if not docs:
        return "📚 引用来源：未检索到相关文档"

    sources = []
    seen = set()

    for doc in docs:
        file_name = os.path.basename(doc.metadata.get("source", "未知文档"))
        page = doc.metadata.get("page", None)

        # page=0 表示第1页，不能用 if page 判断
        if page is not None:
            source = f"{file_name}（第{page + 1}页）"
        else:
            source = file_name

        if source not in seen:
            seen.add(source)
            sources.append(source)

    source_lines = [
        f"{idx}. {source}"
        for idx, source in enumerate(sources, start=1)
    ]

    return "📚 引用来源：\n" + "\n".join(source_lines)


def build_context(docs):
    """
    将检索出的文档片段拼成上下文。
    """
    context_blocks = []

    for idx, doc in enumerate(docs, start=1):
        file_name = os.path.basename(doc.metadata.get("source", "未知文档"))
        page = doc.metadata.get("page", None)

        if page is not None:
            source = f"{file_name}，第{page + 1}页"
        else:
            source = file_name

        block = f"""
【资料{idx}】
来源：{source}
内容：
{doc.page_content}
""".strip()

        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def count_pdf_files(folder):
    """
    统计 PDF 文件数量。
    """
    if not os.path.exists(folder):
        return 0

    return sum(
        1
        for _, _, files in os.walk(folder)
        for file in files
        if file.lower().endswith(".pdf")
    )


# ===================== 全局对象 =====================
embeddings = LocalEmbeddings()
retriever = None
llm = None


# ===================== 初始化知识库 =====================
def init_knowledge_base():
    global retriever, llm

    # 1. 检查 API Key 和模型 ID
    if not API_KEY.strip():
        print("❌ 请先填写 API_KEY")
        return False

    if not LLM_MODEL_ID.strip():
        print("❌ 请先填写 LLM_MODEL_ID")
        return False

    # 2. 检查 docs 文件夹
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        print(f"📁 已创建 docs 文件夹：{DOCS_FOLDER}")
        print("请把 PDF 放进去后重新运行。")
        return False

    pdf_count = count_pdf_files(DOCS_FOLDER)

    if pdf_count == 0:
        print(f"❌ {DOCS_FOLDER} 中没有找到 PDF 文件")
        return False

    # 3. 若要求重建，则删除旧向量库
    if REBUILD_VECTOR_DB and os.path.exists(VECTOR_DB_PATH):
        print("♻️ 检测到 REBUILD_VECTOR_DB=True，正在删除旧向量库...")
        shutil.rmtree(VECTOR_DB_PATH)
        print("✅ 旧向量库已删除")

    # 4. 加载或重建向量库
    if os.path.exists(VECTOR_DB_PATH):
        print("✅ 检测到已存在向量库，正在加载...")

        db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings
        )

    else:
        print("📄 未检测到向量库，开始加载 docs 文件夹中的 PDF...")

        loader = DirectoryLoader(
            DOCS_FOLDER,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            loader_kwargs={"mode": "page"},
            show_progress=True
        )

        docs = loader.load()

        if not docs:
            print("❌ PDF 文档加载失败")
            return False

        print(f"\nPDF 文件数：{pdf_count}")
        print(f"按页加载后的文档单元数：{len(docs)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""]
        )

        splits = text_splitter.split_documents(docs)

        if not splits:
            print("❌ 文本切分结果为空")
            return False

        print(f"切分后的文本块数：{len(splits)}")
        print("🧠 正在生成本地向量知识库...")

        db = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )

        print("✅ 向量库构建并保存完成！")

    # 5. 初始化检索器
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    # 6. 初始化火山方舟豆包 LLM
    llm = ChatOpenAI(
        model=LLM_MODEL_ID,
        temperature=0.1,
        api_key=API_KEY,
        base_url="https://ark.cn-beijing.volces.com/api/v3"
    )

    print("🚀 系统初始化完成，可以开始提问。")
    return True


# ===================== 文档问答 =====================
def chat_response(message, chat_history):
    global retriever, llm

    chat_history = chat_history or []

    if not message or not message.strip():
        return "", chat_history

    if retriever is None or llm is None:
        reply = "系统尚未完成初始化，请检查 API Key、模型 ID 或知识库配置。"

        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})

        return "", chat_history

    try:
        # 1. 检索相关文档
        relevant_docs = retriever.invoke(message)

        # 2. 无召回内容
        if not relevant_docs:
            reply = (
                "抱歉，我在当前文档中没有检索到与该问题足够相关的内容。\n\n"
                "📚 引用来源：未检索到相关文档"
            )

            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": reply})

            return "", chat_history

        # 3. 构造上下文与来源
        context = build_context(relevant_docs)
        source_text = get_source_text(relevant_docs)

        # 4. 整理最近对话历史
        recent_history = chat_history[-MAX_HISTORY_MESSAGES:]

        history_text = ""
        for msg in recent_history:
            role = "用户" if msg["role"] == "user" else "助手"
            history_text += f"{role}：{msg['content']}\n"

        # 5. Prompt
        prompt = f"""
你是一个面向气象论文、雷达资料和专业报告的 PDF 智能问答助手。

请严格遵守以下规则：
1. 只能依据“参考资料”中的内容回答。
2. 不允许凭空补充文档中没有的信息。
3. 若参考资料不足以回答，直接说明：“抱歉，我在文档中没有找到足够明确的相关内容。”
4. 回答应尽量清晰、准确，优先使用分点表达。
5. 不要伪造引用，不要虚构页码。
6. 若问题涉及多个资料，可综合归纳。
7. 保持气象与雷达相关术语准确。

【历史对话】
{history_text}

【参考资料】
{context}

【用户问题】
{message}

请给出回答：
"""

        # 6. 调用豆包 LLM
        res = llm.invoke(prompt)
        answer = res.content.strip()

        reply = answer + "\n\n" + source_text

        # 7. 更新历史
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})

        return "", chat_history

    except Exception as e:
        error_reply = f"系统处理异常：{str(e)}"

        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_reply})

        return "", chat_history


# ===================== 结构化信息抽取 =====================
def extract_structured_info(topic):
    global retriever, llm

    if not topic or not topic.strip():
        return "请输入需要抽取的气象主题或关键词。"

    if retriever is None or llm is None:
        return "系统尚未完成初始化，请检查配置。"

    try:
        # 1. 构造更适合抽取任务的检索 query
        search_query = f"""
        {topic}
        研究对象 数据资料 研究方法 技术路线
        结果分析 核心结论 关键知识点
        """

        relevant_docs = retriever.invoke(search_query)

        if not relevant_docs:
            return (
                "抱歉，我在当前文档中没有检索到足够相关的内容。\n\n"
                "📚 引用来源：未检索到相关文档"
            )

        # 2. 上下文和引用
        context = build_context(relevant_docs)
        source_text = get_source_text(relevant_docs)

        # 3. Prompt
        prompt = f"""
你是气象领域论文与技术资料的结构化信息抽取助手。

请严格依据“参考资料”完成抽取，不允许编造。
如果某项信息在资料中没有明确出现，请写“文档未明确说明”。

请围绕用户主题，提取以下内容：

1. 研究主题
2. 研究对象 / 天气过程 / 研究场景
3. 使用的数据资料或观测资料
4. 研究方法 / 技术路线
5. 主要分析内容
6. 核心结论
7. 可沉淀为知识库的关键知识点（列出3条）
8. 100字以内的简要总结

输出要求：
- 使用清晰小标题
- 每项内容分开写
- 保持气象和雷达专业术语准确
- 只能依据参考资料回答

【用户主题】
{topic}

【参考资料】
{context}

请开始抽取：
"""

        res = llm.invoke(prompt)
        result = res.content.strip()

        return result + "\n\n" + source_text

    except Exception as e:
        return f"结构化抽取异常：{str(e)}"


# ===================== 清空对话 =====================
def clear_chat():
    return []


# ===================== 前端界面 =====================
def create_web_ui():
    with gr.Blocks(title="气象资料智能问答系统") as demo:
        gr.Markdown("# 📖 气象资料智能问答系统（火山方舟豆包 LLM + 本地向量版）")
        gr.Markdown(
            """
            **系统功能：**
            - 自动加载 `docs` 文件夹中的 PDF
            - 基于本地向量检索实现 RAG 文档问答
            - 使用火山方舟豆包大模型生成回答
            - 支持多轮对话
            - 自动显示引用文件与页码
            - 支持论文结构化信息抽取
            """
        )

        # ==================== Tab 1：文档问答 ====================
        with gr.Tab("💬 文档智能问答"):
            chatbot = gr.Chatbot(
                height=520,
                label="问答窗口"
            )

            msg = gr.Textbox(
                placeholder="例如：弱垂直风切变环境下，强下击暴流有哪些双偏振雷达特征？",
                label="问题输入"
            )

            clear = gr.Button("清空对话")

            msg.submit(
                chat_response,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )

            clear.click(
                clear_chat,
                inputs=[],
                outputs=chatbot
            )

        # ==================== Tab 2：结构化信息抽取 ====================
        with gr.Tab("🧾 结构化信息抽取"):
            gr.Markdown(
                """
                输入一个气象主题，系统会从 PDF 知识库中检索相关资料，
                并抽取研究主题、对象、方法、结论与关键知识点。
                """
            )

            extract_input = gr.Textbox(
                label="输入主题",
                placeholder="例如：弱垂直风切变环境下强下击暴流",
                lines=2
            )

            extract_btn = gr.Button("开始结构化抽取")

            extract_output = gr.Textbox(
                label="抽取结果",
                lines=28
            )

            extract_btn.click(
                extract_structured_info,
                inputs=extract_input,
                outputs=extract_output
            )

    demo.launch(inbrowser=True)


# ===================== 主程序入口 =====================
if __name__ == "__main__":
    if init_knowledge_base():
        create_web_ui()