# ==============================================
# 实时 RAG 系统（自动监控文件 → 自动更新知识库）
# 基于 Ollama + llama3.2:1b + Chroma + 实时监控
# 新增文档 → 自动向量化 → 立即可用
# ==============================================
import os
import time
import ollama
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===================== 配置 =====================
VECTOR_DB_PATH = "./real_time_rag_db"
DOCS_FOLDER = "./my_docs"  # 把文档放这里，自动实时更新
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
LLM_MODEL = "llama3.2:1b"

# 自动创建文档目录
os.makedirs(DOCS_FOLDER, exist_ok=True)

# ===================== 加载模型 =====================
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}# 关键：提高检索精度
      ) 
splitter = RecursiveCharacterTextSplitter(
    chunk_size=384, 
    chunk_overlap=64,   
    separators=["\n\n", "\n", "。", "！", "？", ""]
)

# ===================== 向量库 =====================
def get_vector_db():
    if os.path.exists(VECTOR_DB_PATH):
        return Chroma(embedding_function=embeddings, persist_directory=VECTOR_DB_PATH)
    return Chroma(embedding_function=embeddings, persist_directory=VECTOR_DB_PATH)

# ===================== 加载单个文档（TXT/PDF） =====================
def load_single_file(file_path):
    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            print(f"📄 加载 PDF 文档：{os.path.basename(file_path)}")
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            print(f"⚠️ 不支持的文件格式：{os.path.basename(file_path)}")
            return []

        docs = loader.load()
        print (f"✅ 成功加载文档：{os.path.basename(file_path)}，共 {len(docs)} 页")
        return splitter.split_documents(docs)
    except Exception as e  :
        print("异常：", e)
        print(f"❌ 加载文档失败：{os.path.basename(file_path)}，错误：{str(e)}")    
        print(f"⚠️ 无法处理文件：{file_path}，请确保是 PDF 或 TXT 格式")
        return []
# ===================== 首次启动加载所有文档 =====================
def load_all_existing_files():
    print("\n📂 正在加载已有文档...")
    for f in os.listdir(DOCS_FOLDER):
        print(f"\n📂 正在加载已有文档：{f}")
        path = os.path.join(DOCS_FOLDER, f)
        load_single_file(path)

# ===================== 增量添加到向量库（实时更新） =====================
def add_file_to_db(file_path):
    chunks = load_single_file(file_path)
    if not chunks:
        return
    vectordb = get_vector_db()
    vectordb.add_documents(chunks)
    print(f"✅ 已实时更新新文档：{os.path.basename(file_path)}")

# ===================== 监控文件夹变化 =====================
class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            time.sleep(0.5)
            add_file_to_db(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            time.sleep(0.5)
            add_file_to_db(event.src_path)

# ===================== 启动监控 =====================
def start_monitor():
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, DOCS_FOLDER, recursive=False)
    observer.start()
    print("📂 实时监控已启动：新增/修改文档会自动更新 RAG 知识库")

# ===================== RAG 问答（永远用最新资料） =====================
def rag_ask(question):
    vectordb = get_vector_db()
    docs = vectordb.similarity_search(question, k=4)

    # 调试：打印真正检索到的内容
    print("\n====== 系统找到的相关资料 ======")
    for i, d in enumerate(docs):
        print(f"{i+1}. {d.page_content[:80]}...")
    print("================================\n")

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
你是严格的资料问答助手，必须遵守以下规则：
1. 只使用下方提供的资料回答
2. 绝对不能编造内容
3. 不知道就说：资料中未提及
4. 回答简洁、准确、忠于原文


资料：
{context}

问题：{question
回答：
"""
    res = ollama.generate(model=LLM_MODEL, prompt=prompt)
    return res["response"]

# ===================== 启动 =====================
if __name__ == "__main__":
    print("正在初始化 RAG 系统...")
    load_all_existing_files()
    start_monitor()

    print("=" * 60)
    print("    🚀 实时 RAG 已启动！")
    print(f"    文档目录：{DOCS_FOLDER}")
    print("    放入文件 → 自动更新知识库")
    print("=" * 60)

    while True:
        q = input("\n请提问：")
        if q.lower() == "quit":
            break
        print("AI 思考中...")
        ans = rag_ask(q)
        print("\n💡 回答：\n", ans)