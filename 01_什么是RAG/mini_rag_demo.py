# mini_rag_demo.py
# 最小可运行的 RAG 示例 - 演示 RAG 的核心工作流程
# 不依赖向量数据库，用简单的关键词匹配代替向量检索

# ===== 第一步：准备知识库 =====
knowledge_base = [
    "RAG 全称是 Retrieval-Augmented Generation，即检索增强生成。",
    "RAG 由 Facebook 在 2020 年提出，核心思想是让大模型在生成回答前先检索相关文档。",
    "RAG 的主要优势是可以避免大模型产生幻觉，同时知识可以实时更新。",
    "向量数据库是 RAG 系统的核心组件之一，用于存储和检索文档的向量表示。",
    "Embedding 是将文本转换为向量的技术，是 RAG 的基础。",
]

# ===== 第二步：检索（简化版，用关键词匹配代替向量检索） =====
def simple_search(query, documents, top_k=3):
    """最简单的检索：计算 query 和每个文档的重叠词数量"""
    query_words = set(query.lower().split())
    scored = []
    for doc in documents:
        doc_words = set(doc.lower().split())
        score = len(query_words & doc_words)  # 交集大小作为相似度
        scored.append((score, doc))
    scored.sort(reverse=True)
    return [doc for score, doc in scored[:top_k]]

# ===== 第三步：构建 Prompt 并调用大模型 =====
def build_prompt(query, retrieved_docs):
    prompt = f"""你是一个问答助手。请根据以下参考资料回答用户问题。
如果参考资料中没有相关信息，请如实说明。

参考资料：
{chr(10).join(f'- {doc}' for doc in retrieved_docs)}

用户问题：{query}

请根据参考资料回答："""
    return prompt

# ===== 完整流程 =====
if __name__ == "__main__":
    user_query = "RAG 是什么？它有什么好处？"

    # 1. 检索
    retrieved = simple_search(user_query, knowledge_base, top_k=3)
    print("=" * 50)
    print("检索到的相关文档：")
    for i, doc in enumerate(retrieved, 1):
        print(f"  [{i}] {doc}")

    # 2. 构建 Prompt
    prompt = build_prompt(user_query, retrieved)
    print("\n" + "=" * 50)
    print("构建的 Prompt：")
    print(prompt)

    # 3. 注意：这里没有真正调用大模型 API
    # 实际项目中，你会把 prompt 发送给 OpenAI / 通义千问 / DeepSeek 等
    print("\n" + "=" * 50)
    print("在真实项目中，上面的 Prompt 会被发送给大模型 API 来获取最终回答。")
