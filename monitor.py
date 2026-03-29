import time
import os
from rag import add_file_to_db
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

DOCS_FOLDER = "./my_docs" # 监控这个文件夹，新增/修改文档会自动更新 RAG 知识库

# 自动创建文档目录
os.makedirs(DOCS_FOLDER, exist_ok=True)

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