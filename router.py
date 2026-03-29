import ollama


LOCAL_MODEL = "qwen:4b"          # 本地模型


# 判断是否需要调用API（简单问题本地，复杂问题API）
def need_rag_or_api(question):
    q = question.strip().lower()
    
    # 规则：包含这些关键词 = 复杂问题 = 用API
    keywords = ["什么", "怎么", "如何", "为什么", "多少", "谁", "哪里", "几时", "介绍", "说明",
                "文件", "文档", "资料", "内容", "讲了", "提到", "告诉我", "总结"]
    
    # 长度>15个字 = 复杂问题 = API
    if len(question) > 15:
        return True
    
    # 包含关键词 = API
    for kw in keywords:
        if kw in q:
            return True
    
    # 否则 = 本地
    return False