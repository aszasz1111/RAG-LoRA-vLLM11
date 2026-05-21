from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from dashscope import TextEmbedding
import os
import warnings
import gradio as gr

# 忽略警告
warnings.filterwarnings("ignore")

# ===================== 配置区（只改这里！）=====================
API_KEY = "sk-da6ab4fb9490463a8c1e32617b0e2a5c"
DOCS_FOLDER = "./docs"
VECTOR_DB_PATH = "./chroma_db"
# ==============================================================

# 全局配置密钥
os.environ["DASHSCOPE_API_KEY"] = API_KEY
dashscope.api_key = API_KEY

# 自定义嵌入模型（带错误捕获）
class QwenEmbeddings:
    def embed_documents(self, texts):
        try:
            resp = TextEmbedding.call(
                model=TextEmbedding.Models.text_embedding_v1,
                input=texts
            )
            if resp.status_code != 200:
                raise Exception(f"嵌入模型调用失败：{resp.message}")
            return [item["embedding"] for item in resp.output["embeddings"]]
        except Exception as e:
            print(f"嵌入模型错误：{e}")
            raise

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# 全局变量
embeddings = QwenEmbeddings()
retriever = None
llm = None

# 初始化知识库（带日志）
def init_knowledge_base():
    global retriever, llm
    print("=== 开始初始化知识库 ===")

    # 1. 加载/创建向量库
    if os.path.exists(VECTOR_DB_PATH):
        print("✅ 检测到已存在向量库，正在加载...")
        db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings
        )
    else:
        print("⚠️ 未检测到向量库，开始加载PDF...")
        if not os.path.exists(DOCS_FOLDER):
            os.makedirs(DOCS_FOLDER)
            print(f"❌ 请将PDF文件放入 {DOCS_FOLDER} 文件夹后重启程序")
            return False

        # 加载PDF
        loader = DirectoryLoader(
            DOCS_FOLDER,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        docs = loader.load()
        print(f"✅ 加载了 {len(docs)} 个PDF文件")

        if len(docs) == 0:
            print("❌ docs文件夹中没有找到PDF文件，请检查文件格式")
            return False

        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(docs)
        print(f"✅ 分割为 {len(splits)} 个文本块")

        # 创建向量库
        print("⏳ 正在生成向量库...")
        db = Chroma.from_documents(
            splits,
            embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        print("✅ 向量库创建完成！")

    # 2. 初始化检索器
    retriever = db.as_retriever(search_kwargs={"k": 3})
    print("✅ 检索器初始化完成")

    # 3. 初始化LLM
    llm = ChatOpenAI(
        model="qwen-turbo",
        temperature=0.1,
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    print("✅ LLM初始化完成")
    print("=== 知识库初始化完成 ===")
    return True

# 对话函数（带错误捕获）
def chat_response(message, chat_history):
    if not message.strip():
        return "", chat_history

    try:
        print(f"用户提问：{message}")
        # 1. 检索文档
        relevant_docs = retriever.invoke(message)
        print(f"✅ 检索到 {len(relevant_docs)} 个相关文档")

        context = "\n".join([d.page_content for d in relevant_docs])
        sources = [os.path.basename(d.metadata["source"]) for d in relevant_docs]
        source_text = "📚 引用来源：" + "、".join(sources)

        # 2. 构造prompt
        prompt = f"""
你是气象资料问答助手，请严格根据上下文回答问题，不要编造。
上下文：{context}
问题：{message}
"""
        # 3. 调用LLM
        res = llm.invoke(prompt)
        reply = res.content + "\n\n" + source_text

        # 4. 更新对话历史
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        print("✅ 回答生成完成")
        return "", chat_history

    except Exception as e:
        print(f"❌ 对话错误：{e}")
        error_msg = f"系统错误：{str(e)}"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_msg})
        return "", chat_history

# 清空对话
def clear_chat():
    return []

# 创建界面（适配新版Gradio）
def create_web_ui():
    with gr.Blocks(title="气象资料智能问答系统") as demo:
        gr.Markdown("# 📖 气象资料智能问答系统（RAG）")
        gr.Markdown("自动加载docs文件夹PDF | 多轮对话 | 显示引用来源")

        chatbot = gr.Chatbot(height=500, type="messages")
        msg = gr.Textbox(placeholder="输入你的问题...")
        clear_btn = gr.Button("清空对话")

        # 绑定事件
        msg.submit(chat_response, [msg, chatbot], [msg, chatbot])
        clear_btn.click(clear_chat, [], chatbot)

    return demo

if __name__ == "__main__":
    # 先初始化知识库
    if init_knowledge_base():
        # 启动界面
        demo = create_web_ui()
        demo.launch(inbrowser=True, debug=True)