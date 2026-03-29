# ==============================================
# 实时 RAG 系统（自动监控文件 → 自动更新知识库）
# 基于 Ollama + llama3.2:1b + Chroma + 实时监控
# 新增文档 → 自动向量化 → 立即可用
# ==============================================
import os
import time
from flask import Flask, render_template, request, jsonify
from rag import get_vector_db, load_all_existing_files,rag_ask,add_file_to_db
from monitor import start_monitor

# ===================== 配置 =====================
app = Flask(__name__, template_folder="static", static_folder="static")
# app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./my_docs"
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# ===================== 网页路由 =====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return "", 204

@app.route('/api/chat', methods=['POST'])
def chat():
    request_data = request.get_json(force=True)
    question = request_data.get('question')
    return jsonify({"answer": rag_ask(question)})

@app.route('/api/files')
def list_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return jsonify({"files": files})

@app.route('/api/upload', methods=['POST'])
def upload():
    file = request.files['file']
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    add_file_to_db(path)
    return jsonify({"status": "ok"})

@app.route('/api/delete/<filename>')
def delete_file(filename):
    try:
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # clear_db()
        load_all_existing_files()
        return jsonify({"status": "ok"})
    except:
        return jsonify({"status": "error"})

@app.route('/api/clear-db')
def clear_database():
    db = get_vector_db()
    db.delete_collection()  # 删除整个集合，清空库
    return jsonify({"status": "ok"})

@app.route('/api/init-db')
def init_database():
    load_all_existing_files()
    return jsonify({"status": "ok"})

# ===================== 【安全查看向量库内容】无报错版 =====================
@app.route('/api/print-all-data')
def print_all_data():
    try:
        db = get_vector_db()
        
        # ✅ 只获取 文本、元数据，不获取向量（修复报错）
        all_data = db._collection.get(
            include=["documents", "metadatas"]
        )

        print("\n" + "="*60)
        print("📂 向量库 real_time_rag_db 真实内容")
        print("="*60)

        result = []
        for i in range(len(all_data["ids"])):
            doc = all_data["documents"][i]
            meta = all_data["metadatas"][i]
            doc_id = all_data["ids"][i]

            print(f"\n【第 {i+1} 条】")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print(f"ID: {doc_id}")
            print(f"来源: {meta}")
            print(f"内容: {doc[:100]}...")
            print("-"*50)

            result.append({
                "id": doc_id,
                "source": meta.get("source", "未知文件"),
                "content": doc[:150]  # 只显示前150字符
            })

        return jsonify({
            "total": len(result),
            "items": result
        })

    except Exception as e:
        print("查看向量库错误：", e)
        return jsonify({"error": str(e)})
# ===================== 启动 =====================
if __name__ == '__main__':
    load_all_existing_files()
    start_monitor()
    print("=" * 60)
    print("    🌍 网页访问：http://127.0.0.1:8080")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8080, debug=True)