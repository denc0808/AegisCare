// 消息显示
function appendMsg(text, isUser = false) {
    const box = document.getElementById('msg');
    if (!box) return;

    const div = document.createElement('div');
    div.className = isUser ? 'msg-user' : 'msg-bot';
    div.innerText = text;
    box.appendChild(div);
    div.scrollIntoView({ behavior: 'smooth' });
}

// 发送消息
async function sendMsg() {
    const input = document.getElementById('question');
    const q = input.value.trim();
    if (!q) return;

    appendMsg(q, true);
    input.value = '';

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: q })
        });
        const data = await res.json();
        appendMsg(data.answer);
    } catch (e) {
        appendMsg("请求失败");
    }
}

// 加载文件列表
async function loadFiles() {
    const res = await fetch('/api/files');
    const data = await res.json();
    const list = document.getElementById('file-list');
    list.innerHTML = '';

    data.files.forEach(f => {
        const item = document.createElement('div');
        item.className = 'file-item';
        item.innerHTML = `<span>${f}</span><button class='btn btn-sm btn-danger' onclick='delFile("${f}")'>删除</button>`;
        list.appendChild(item);
    });
}

// 上传
async function doUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);
    await fetch('/api/upload', { method: 'POST', body: fd });
    loadFiles();
    e.target.value = '';
}

// 删除
async function delFile(filename) {
    if (!confirm('确定删除？')) return;
    await fetch(`/api/delete/${filename}`);
    loadFiles();
}

// 清空库
async function clearDB() {
    if (!confirm('确定清空？')) return;
    await fetch('/api/clear-db');
    loadFiles();
}

// 清空库
async function initDB() {
    if (!confirm('确定初始化？')) return;
    await fetch('/api/init-db');
}

// 弹窗 - 查看向量库
async function openVectorModal() {
    const modal = document.getElementById('vectorModal');
    const content = document.getElementById('vectorModalContent');
    const res = await fetch('/api/print-all-data');
    const data = await res.json();

    if (data.error) {
        content.innerHTML = '<p>暂无数据</p>';
    } else {
        let html = `总条数：${data.total}<br><br>`;
        data.items.forEach((item, i) => {
            html += `<div style='padding:8px; background:#333; margin:5px 0; border-radius:6px;'>
                #${i + 1} <br>
                来源：${item.source} <br>
                内容：${item.content}
            </div>`;
        });
        content.innerHTML = html;
    }
    modal.style.display = 'block';
}

function closeVectorModal() {
    document.getElementById('vectorModal').style.display = 'none';
}

// 🔥 按 ESC 关闭弹窗（干净无 static）
document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") {
        closeVectorModal();
    }
});

// 回车发送
document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('question');
    input.addEventListener('keydown', e => {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendMsg();
        }
    });
});

window.onload = loadFiles;