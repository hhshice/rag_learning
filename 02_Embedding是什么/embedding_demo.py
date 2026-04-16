# embedding_demo.py
# Embedding 原理演示 - 字符级相似度（无需任何依赖）
# 无需 API Key，通过简单算法直观理解 Embedding 的工作原理

def simple_tokenize(text):
    """简单分词：按字符拆分"""
    return set(text.lower())

def cosine_similarity(a, b):
    """计算两个集合的 Jaccard 相似度（简化版）"""
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0

def main():
    print("=" * 60)
    print("Embedding 原理演示 - 字符级相似度")
    print("=" * 60)

    # ===== 文档知识库 =====
    documents = [
        "如何修理笔记本电脑",
        "汽车发动机故障排查",
        "台式机内存升级教程",
        "餐厅美食推荐",
        "Python编程入门",
        "机器学习算法原理",
        "电脑屏幕不显示",
        "川菜麻辣菜谱"
    ]

    # 用户查询
    query = "电脑坏了怎么修"

    print(f"\n📝 知识库文档数量: {len(documents)}")
    print(f"🔍 用户查询: {query}\n")

    # ===== 查询分词 =====
    query_chars = simple_tokenize(query)
    print(f"📌 查询字符集合: {query_chars}\n")

    # ===== 计算相似度 =====
    similarities = []
    for doc in documents:
        doc_chars = simple_tokenize(doc)
        sim = cosine_similarity(query_chars, doc_chars)
        similarities.append((sim, doc, doc_chars))

    # 按相似度排序
    similarities.sort(reverse=True)

    print("-" * 60)
    print("📊 相似度排序结果：\n")

    for i, (sim, doc, chars) in enumerate(similarities, 1):
        common = query_chars & chars
        bar_len = int(sim * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  [{i}] {bar} {sim:.4f}  {doc}")
        print(f"      公共字符: {common}")

    print("\n" + "=" * 60)
    print("✅ 关键发现：")
    top = similarities[0]
    print(f"   '{query}' 与 '{top[1]}' 最相似")
    print(f"   公共字符: {top[2] & query_chars}")
    print("=" * 60)

    print("\n📌 Embedding 的核心思想：")
    print("   1. 把文字转换成向量（一串数字）")
    print("   2. 语义相似的文字 → 向量也相似")
    print("   3. 通过计算向量距离判断语义相似度")

    print("\n📌 本示例用字符重叠模拟 Embedding（简化版）")
    print("   真正的 Embedding 用深度学习模型（如 BERT）")

if __name__ == "__main__":
    main()
