import os
import requests
import hashlib
import ollama
import shutil
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from router import need_rag_or_api

# ===================== 配置 =====================
VECTOR_DB_PATH = "./real_time_rag_db"
DOCS_FOLDER = "./my_docs"  # 把文档放这里，自动实时更新
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
LLM_MODEL = "qwen:4b"

API_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # API模型
API_KEY = "sk-zfchmunzqyczprffzpnrjayemtyarrghloihkwivpvzuukjr"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# 自动创建文档目录
os.makedirs(DOCS_FOLDER, exist_ok=True)

# ===================== 加载模型 =====================
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}# 关键：提高检索精度
      ) 
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,  # 增大块大小，减少碎片化
    chunk_overlap=64,   
    separators=["\n\n", "\n", "。", "！", "？", ""]
)


def get_vector_db():
    try:
        if os.path.exists(VECTOR_DB_PATH):
            return Chroma(
                    embedding_function=embeddings,
                    persist_directory=VECTOR_DB_PATH
            )
        return Chroma(embedding_function=embeddings, persist_directory=VECTOR_DB_PATH)
    except:
        # 失败了 = 重建全新数据库
        clear_db()
        return Chroma(
            embedding_function=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
    

def clear_db():
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)
        return True
    return False

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
    
def add_file_to_db(file_path):
    chunks = load_single_file(file_path)
    if not chunks:
        return

    vectordb = get_vector_db()
    filename = os.path.basename(file_path)

    # 生成唯一 ID：文件名 + 内容哈希 = 绝对不重复
    ids = []
    for chunk in chunks:
        content = chunk.page_content.strip()
        hash_str = hashlib.md5(content.encode('utf-8')).hexdigest()[:12]
        unique_id = f"{filename}_{hash_str}"
        ids.append(unique_id)

    # 🔥 关键：用 ids 添加，重复 ID 不会重复入库
    vectordb.add_documents(documents=chunks, ids=ids)
    print(f"✅ 去重入库：{filename} | 块数：{len(chunks)}")
    print(f"✅ 已实时更新新文档：{os.path.basename(file_path)}")
    vectordb.persist()
# ===================== 首次启动加载所有文档 =====================
def load_all_existing_files():
    print("\n重建向量数据库")
    try:
        vectordb = get_vector_db()
        # vectordb.delete_collection()  # 删除旧数据，重新构建
        print("\n📂 正在加载已有文档...")
        for f in os.listdir(DOCS_FOLDER):
            print(f"\n📂 正在加载已有文档：{f}")
            path = os.path.join(DOCS_FOLDER, f)
            chunks = load_single_file(path)
            # ✅【关键修复】把分块后的数据存入向量库
            if chunks:
                vectordb.add_documents(chunks)
                print(f"✅ 成功存入向量库：{f}，块数：{len(chunks)}")

        vectordb.persist()
    except:
        clear_db()

# ===================== 增量添加到向量库（实时更新） =====================
# def add_file_to_db(file_path):
#     chunks = load_single_file(file_path)
#     if not chunks:
#         print("❌ 没有可添加的内容")
#         return
#     vectordb = get_vector_db()
#     vectordb.add_documents(chunks)
#     print(f"✅ 已实时更新新文档：{os.path.basename(file_path)}")
#     vectordb.persist()
    
    

# ===================== RAG 问答（永远用最新资料） =====================
def rag_ask(question):

    if not question or not question.strip():
        return "请输入有效的问题。"
    use_api = need_rag_or_api(question)

    vectordb = get_vector_db()
    docs = vectordb.similarity_search(question, k=4)

    # 调试：打印真正检索到的内容
    unique_contents = []
    seen = set()
    for d in docs:
        content = d.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique_contents.append(d)
    docs = unique_contents

    print("\n====== 系统找到的相关资料 ======")
    for i, d in enumerate(docs):
        print(f"{i+1}. {d.page_content[:80]}...")
    print("================================\n")

    # context = "\n".join([d.page_content for d in docs])
    valid_docs = [d for d in docs if d.page_content and str(d.page_content).strip()]

    context = "\n---\n".join([str(d.page_content).strip() for d in valid_docs])
    prompt = f"""
你是一个严格的资料问答机器人，必须遵守以下所有规则：
1. 只使用下面的【资料内容】回答，绝对不能使用外部知识。


资料：
{context}

问题：{question}
回答：
""" 
    if  use_api:
        print("🤖 使用API模型（复杂问题）")
        res =  api_llm(prompt)
    else:
        print("⚡ 使用本地模型（简单问题）")
        res = local_llm(prompt)
    return res


def local_llm(prompt):
    try:
        res = ollama.generate(model=LLM_MODEL, prompt=prompt,options={"temperature": 0.1})
        return res.get("response", "本地模型失败")
    except:
        return "本地模型加载中，请稍候..."
    

def api_llm(prompt):
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": API_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        resp = requests.post(API_URL, json=data, headers=headers, timeout=20)
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("API错误：", e)
        return "API调用失败，改用本地模型\n" + local_llm(prompt)