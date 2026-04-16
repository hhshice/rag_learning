# faiss_demo.py
# 使用 FAISS 进行高效向量检索

# 安装依赖：pip install faiss-cpu sentence-transformers

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def main():
    print("=" * 60)
    print("⚡ FAISS 高效向量检索示例")
    print("=" * 60)
    
    # 1. 加载 Embedding 模型
    print("\n📦 加载 Embedding 模型...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ 模型加载完成")
    
    # 2. 准备文档
    documents = [
        "RAG 是检索增强生成技术。",
        "向量数据库用于存储向量。",
        "Python 是一门编程语言。",
        "深度学习使用神经网络。",
        "FastAPI 是 Python Web 框架。",
        "机器学习是人工智能的分支。",
        "自然语言处理处理人类语言。",
        "计算机视觉让机器看懂图像。"
    ]
    
    # 3. 生成向量
    print("\n🔄 生成文档向量...")
    embeddings = model.encode(documents)
    embeddings = np.array(embeddings).astype('float32')
    
    print(f"✅ 向量维度：{embeddings.shape}")
    
    # 4. 构建 FAISS 索引
    print("\n🔨 构建 FAISS 索引...")
    dimension = embeddings.shape[1]  # 向量维度
    
    # 方法 1: 简单的暴力搜索（IndexFlatL2）
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print(f"✅ FAISS 索引构建完成，包含 {index.ntotal} 个向量")
    
    # 5. 检索
    queries = [
        "什么是 RAG？",
        "编程语言有哪些？",
        "AI 相关技术"
    ]
    
    print("\n" + "=" * 60)
    print("🔍 执行检索")
    print("=" * 60)
    
    for query in queries:
        print(f"\n查询：{query}")
        print("-" * 60)
        
        query_vector = model.encode([query]).astype('float32')
        
        k = 3  # 返回 top-k
        distances, indices = index.search(query_vector, k)
        
        for i, idx in enumerate(indices[0]):
            similarity = 1 / (1 + distances[0][i])  # 将距离转换为相似度
            print(f"  [{i+1}] 相似度: {similarity:.4f}")
            print(f"      {documents[idx]}")
    
    # 6. 演示更高效的索引（IVF）
    print("\n" + "=" * 60)
    print("⚡ 高级索引：IVF（倒排文件索引）")
    print("=" * 60)
    
    # 创建 IVF 索引
    nlist = 2  # 聚类中心数量
    quantizer = faiss.IndexFlatL2(dimension)
    index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    
    # 训练索引（需要先训练聚类中心）
    print("\n🏋️ 训练索引...")
    index_ivf.train(embeddings)
    index_ivf.add(embeddings)
    
    print(f"✅ IVF 索引构建完成")
    
    # 使用 IVF 索引检索
    query = "Python 技术"
    query_vector = model.encode([query]).astype('float32')
    
    k = 3
    distances, indices = index_ivf.search(query_vector, k)
    
    print(f"\n查询：{query}")
    print("-" * 60)
    for i, idx in enumerate(indices[0]):
        print(f"  [{i+1}] {documents[idx]}")
    
    print("\n" + "=" * 60)
    print("✅ FAISS 演示完成！")
    print("=" * 60)
    print("\n💡 提示：")
    print("   - IndexFlatL2：暴力搜索，精度高但速度慢")
    print("   - IndexIVFFlat：聚类索引，速度快但需要训练")
    print("   - 对于海量数据（百万级），IVF 速度优势明显")

if __name__ == "__main__":
    main()
