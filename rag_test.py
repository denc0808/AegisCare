# ==============================================
# 本地 RAG 项目
# 基于：Ollama + llama3.2:1b + Chroma 向量库
# 100% 本地运行 | 无报错 | 企业级
# ==============================================

import ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

# ===================== 配置 =====================
VECTOR_DB_PATH   = "./local_rag_db"    # 向量库保存位置
# EMBEDDING_MODEL  = "BAAI/bge-small-zh-v1.5"  # 中文向量化模型
EMBEDDING_MODEL  = "text2vec-base-chinese"  # 中文向量化模型
LLM_MODEL        = "qwen:4b"       # 你要的模型

# ===================== 【你的本地知识库】 =====================
# 这里可以换成你的 PDF / Word / 文档
KNOWLEDGE_BASE = """
员工手册：
1. 工作满1年不满5年，每年年假5天。
2. 工作满5年不满10年，年假10天。
3. 工作满10年及以上，年假15天。
4. 每月最多3天带薪病假，需要医院证明。
5. 工作日加班1.5倍工资，周末2倍，法定节假日3倍。
6. 加班可以调休，比例1:1。
"""

# ===================== 初始化向量库 =====================
def init_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 如果向量库已存在 → 直接加载
    if os.path.exists(VECTOR_DB_PATH):
        print("✅ 加载本地向量数据库...")
        return Chroma(
            embedding_function=embeddings,
            persist_directory=VECTOR_DB_PATH
        )

    # 第一次运行 → 创建向量库
    print("🆕 首次构建本地向量库...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128
    )
    chunks = splitter.split_text(KNOWLEDGE_BASE)
    docs = [Document(page_content=c) for c in chunks]

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    return vectordb

# ===================== 本地 Ollama 调用 =====================
def llm_reply(prompt):
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"]

# ===================== RAG 核心流程 =====================
def ask(question, vectordb):
    print(f"\n🔍 问题：{question}")

    # 1. 从本地向量库检索资料
    docs = vectordb.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])
    print(f"✅ 本地资料：\n{context}\n")

    # 2. 给模型的提示词
    prompt = f"""
你是一个智能助手，请根据资料回答问题，不要编造内容。

资料：
{context}

问题：{question}
回答：
"""

    # 3. 本地模型生成回答
    print("🤖 模型思考中...")
    return llm_reply(prompt)

# ===================== 启动程序 =====================
if __name__ == "__main__":
    vectordb = init_vector_store()

    print("=" * 60)
    print("    🚀 本地 RAG 启动成功")
    print("    模型：llama3.2:1b")
    print("    输入 quit 退出")
    print("=" * 60)

    while True:
        query = input("\n请输入问题：")
        if query.lower() == "quit":
            break
        answer = ask(query, vectordb)
        print("\n" + "=" * 60)
        print("📝 回答：\n", answer)
        print("=" * 60)