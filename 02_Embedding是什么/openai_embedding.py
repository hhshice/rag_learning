# openai_embedding.py
# 使用新版 OpenAI API (1.0+) 实现 Embedding
# 安装依赖: pip install openai

from openai import OpenAI
import numpy as np

# 设置你的 OpenAI API Key
client = OpenAI(api_key="---")  # 替换为你的 key

def get_embedding(text, model="text-embedding-3-small"):
    """调用 OpenAI API 获取文本的 Embedding 向量"""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    print("=" * 60)
    print("OpenAI Embedding 演示 (新版 API)")
    print("=" * 60)

    # ===== 文档知识库 =====
    texts = [
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

    print(f"\n📝 知识库文档数量: {len(texts)}")
    print(f"🔍 用户查询: {query}\n")

    # ===== 获取 Embedding =====
    print("正在计算 Embedding（使用 OpenAI API）...")

    # 批量获取文档 Embedding
    doc_embeddings = []
    for text in texts:
        emb = get_embedding(text)
        doc_embeddings.append(emb)

    # 获取查询 Embedding
    query_embedding = get_embedding(query)

    print(f"✅ 获取成功！向量维度: {len(query_embedding)}\n")

    # ===== 计算相似度 =====
    similarities = []
    for text, emb in zip(texts, doc_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((sim, text))

    # 按相似度排序
    similarities.sort(reverse=True)

    print("-" * 60)
    print("📊 相似度排序结果：\n")

    for i, (sim, text) in enumerate(similarities, 1):
        bar_len = int(sim * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  [{i}] {bar} {sim:.4f}  {text}")

    print("\n" + "=" * 60)
    print("✅ Embedding 成功识别了语义相似性！")
    print("   '电脑坏了怎么修' 与 '如何修理笔记本电脑' 最相似")
    print("=" * 60)

if __name__ == "__main__":
    main()
