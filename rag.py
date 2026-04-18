import hashlib
import json
import os
import re
import shutil

import ollama
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, MessagesState, END

from router import need_rag_or_api

# ===================== 配置 =====================
VECTOR_DB_PATH = "./real_time_rag_db"
DOCS_FOLDER = "./my_docs"  # 把文档放这里，自动实时更新
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
LLM_MODEL = "qwen:4b"

load_dotenv(dotenv_path="./.env", verbose=True, override=True)

API_MODEL = os.getenv("API_MODEL")  # API模型
API_KEY = os.getenv("API_KEY")  # api key
API_URL = os.getenv("API_URL")

# api url  
# 自动创建文档目录
os.makedirs(DOCS_FOLDER, exist_ok=True)

# ===================== 加载模型 =====================
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}  # 关键：提高检索精度
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
        print(f"✅ 成功加载文档：{os.path.basename(file_path)}，共 {len(docs)} 页")
        return splitter.split_documents(docs)
    except Exception as e:
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
            if os.path.exists(DOCS_FOLDER):
                if f.endswith(".pdf") or f.endswith(".txt"):
                    print(f"📄 加载文档：{f}")

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
        print(f"{i + 1}. {d.page_content[:80]}...")
    print("================================\n")

    # context = "\n".join([d.page_content for d in docs])
    valid_docs = [d for d in docs if d.page_content and str(d.page_content).strip()]

    context = "\n---\n".join([str(d.page_content).strip() for d in valid_docs])
    prompt = f"""
你是一个严格的资料问答机器人，必须遵守以下所有规则：
1. 只使用下面的【资料内容】回答，绝对不能使用外部知识。
2. 如果有遇不确定的问题可以问我


资料：
{context}

问题：{question}
回答：
"""
    if use_api:
        print("🤖 使用API模型（复杂问题）")
        res = api_llm(prompt)
    else:
        print("⚡ 使用本地模型（简单问题）")
        res = local_llm(prompt)
    return res


def local_llm(prompt):
    try:
        res = ollama.generate(model=LLM_MODEL, prompt=prompt, options={"temperature": 0.1})
        return res.get("response", "本地模型失败")
    except Exception as e:
        print("本地模型错误：", e)
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
        resp = requests.post(API_URL, json=data, headers=headers, stream=True, timeout=60)
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("API错误：", e)
        return "API调用失败，改用本地模型\n" + local_llm(prompt)

    # ===================== 多引擎全网搜索（C 方案）=====================


# ===================== 最终智能问答 =====================
def rag_answer(question):
    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=question)]
        })
        return result["messages"][-1].content
    except Exception as e:
        print("RAG 问答失败：", e)
        return "系统繁忙：" + str(e)


def clean_web_text(text: str) -> str:
    """清洗搜索结果：去广告、去噪、精简、去重复"""
    if not text:
        return ""

    # 1. 替换空白、换行、多空格
    text = re.sub(r'\s+', ' ', text.strip())

    # 2. 去掉广告/垃圾关键词
    ad_patterns = [
        r'官网', r'广告', r'推广', r'下载', r'app', r'客户端',
        r'立即', r'点击', r'查看更多', r'结果.*:', r'免责声明',
        r'版权所有', r'备案号', r'首页', r'频道', r'>>', r'<<'
    ]
    for p in ad_patterns:
        text = re.sub(p, '', text, flags=re.IGNORECASE)

    # 3. 去掉连续标点、乱标点
    text = re.sub(r'([。，！？、,.])+', r'\1', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9。，！？；：,.\s]', '', text)

    # 4. 去掉太短的垃圾片段
    sentences = [s.strip() for s in text.split(' ') if len(s.strip()) > 6]
    text = ' '.join(sentences)

    # 5. 去重（简单版）
    lines = list(set(text.split('。')))
    lines = [l.strip() for l in lines if l.strip()]
    text = '。'.join(lines[:5])  # 最多保留5句核心

    return text.strip()


def search_web(query: str, engine="baidu", max_results=3):
    """
    多引擎搜索：bing / baidu / google
    返回最新实时信息
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        results = []

        if engine == "bing":
            url = f"https://www.bing.com/search?q={query}"
            res = requests.get(url, headers=headers, timeout=8)
            soup = BeautifulSoup(res.text, "html.parser")
            for g in soup.find_all("li", class_="b_algo")[:max_results]:
                t = g.get_text(strip=True)
                clean = clean_web_text(t)
                if clean:
                    results.append(clean)

        elif engine == "baidu":
            url = f"https://www.baidu.com/s?wd={query}"
            res = requests.get(url, headers=headers, timeout=8)
            soup = BeautifulSoup(res.text, "html.parser")
            for g in soup.find_all("div", class_="result-op")[:max_results]:
                t = g.get_text(strip=True)
                clean = clean_web_text(t)
                if clean:
                    results.append(clean)

        return "\n".join(results) if results else "无搜索结果"
    except Exception as e:
        print("搜索失败", e)
        return "搜索失败"


def llm(prompt):
    if need_rag_or_api(prompt):
        print("🤖 使用API模型（复杂问题）")
        return api_llm(prompt)
    else:
        print("⚡ 使用本地模型（简单问题）")
        return local_llm(prompt)


# ===================== LangGraph 智能决策（A 方案）=====================
# ===================== LangGraph 智能决策（已修复JSON错误）=====================
def decide_strategy(state: MessagesState):
    question = state["messages"][-1].content
    prompt = f"""
问题：{question}
请严格只输出JSON，不要写任何多余文字、解释、标点，只输出以下格式之一：
{{"strategy":"local"}}
{{"strategy":"web"}}
{{"strategy":"both"}}
"""
    try:
        decision = llm(prompt)
        # 🔥 关键修复：强制清洗模型输出，确保一定能解析JSON
        decision = decision.strip()
        # 尝试提取JSON

        json_match = re.search(r'\{.*}', decision, re.DOTALL)
        if json_match:
            decision = json_match.group(0)
        # 测试解析
        json.loads(decision)
    except Exception as e:
        print("决策模型错误：", e)
        # 🔥 兜底策略：绝对不报错
        decision = '{"strategy":"both"}'

    return {"messages": [AIMessage(content=decision)]}


def execute_strategy(state: MessagesState):
    decision = json.loads(state["messages"][-1].content)
    question = state["messages"][-2].content
    strategy = decision.get("strategy")
    print("策略执行:", strategy)

    # 本地检索
    try:
        db = get_vector_db()
        docs = db.similarity_search(question, k=4)
        local_ctx = "\n".join([d.page_content for d in docs if d.page_content.strip()])
    except:
        local_ctx = ""

    # 全网检索
    web_ctx = search_web(question, engine="bing")  # bing / baidu

    # 策略执行
    if strategy == "local":
        ctx = f"【本地】{local_ctx}"
    elif strategy == "web":
        ctx = f"【全网】{web_ctx}"
    else:
        ctx = f"【本地】{local_ctx}\n\n【全网】{web_ctx}"
    print("检索到的信息:", ctx)

    return {"messages": [AIMessage(content=ctx)]}


def generate_final_answer(state: MessagesState):
    ctx = state["messages"][-1].content
    question = state["messages"][-3].content

    prompt = f"""
根据资料回答问题，不要编造。
资料：{ctx}
问题：{question}
回答：
"""
    # resp = client.chat.completions.create(
    #     model=LLM_MODEL,
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.1
    # )
    # return {"messages": [AIMessage(content=resp.choices[0].message.content)]}
    ans = llm(prompt)
    return {"messages": [AIMessage(content=ans)]}


# 构建智能 Agent
def build_rag_web_agent():
    w = StateGraph(MessagesState)
    w.add_node("decide", decide_strategy)
    w.add_node("execute", execute_strategy)
    w.add_node("generate", generate_final_answer)

    w.set_entry_point("decide")
    w.add_edge("decide", "execute")
    w.add_edge("execute", "generate")
    w.add_edge("generate", END)
    return w.compile()


agent = build_rag_web_agent()
