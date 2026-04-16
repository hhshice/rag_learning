# -*- coding: utf-8 -*-
"""
基础检索流程示例
演示从向量数据库检索文档的基本流程

安装依赖：pip install chromadb
"""

import chromadb


def basic_retrieval():
    """
    最基础的检索流程
    
    流程：
    1. 初始化向量数据库
    2. 准备知识库数据
    3. 执行检索
    4. 显示结果
    """
    
    print("=" * 60)
    print("🔍 基础检索流程示例")
    print("=" * 60)
    
    # 1. 初始化向量数据库
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="knowledge_base",
        metadata={"description": "基础知识库"}
    )
    
    print(f"✅ 向量数据库已初始化")
    print(f"   集合名称: {collection.name}")
    print(f"   当前文档数: {collection.count()}")
    
    # 2. 准备知识库数据
    documents = [
        "Python 是一门流行的编程语言，语法简洁，易于学习。",
        "RAG 是检索增强生成技术，让大模型能够基于外部知识回答问题。",
        "向量数据库用于存储和检索文本的向量表示，支持高效的相似度搜索。",
        "Embedding 是将文本转换为向量的技术，是 RAG 的基础。",
        "FastAPI 是一个现代的 Python Web 框架，高性能且易用。",
        "机器学习是人工智能的一个分支，使用算法从数据中学习模式。",
        "深度学习使用多层神经网络进行特征学习和模式识别。",
        "自然语言处理让计算机能够理解和生成人类语言。"
    ]
    
    # 添加文档（如果集合为空）
    if collection.count() == 0:
        collection.upsert(
            ids=[f"doc_{i}" for i in range(len(documents))],
            documents=documents
        )
        print(f"✅ 已添加 {len(documents)} 个文档")
    else:
        print(f"ℹ️  集合已有 {collection.count()} 个文档，跳过添加")
    
    # 3. 执行检索
    print("\n" + "=" * 60)
    print("🔍 执行检索")
    print("=" * 60)
    
    queries = [
        "什么是 RAG？",
        "Python 有什么特点？",
        "如何处理自然语言？"
    ]
    
    for query in queries:
        print(f"\n查询：{query}")
        print("-" * 60)
        
        # 执行向量检索
        results = collection.query(
            query_texts=[query],
            n_results=3  # 返回 top-3
        )
        
        # 显示结果
        for i, (doc_id, document, distance) in enumerate(
            zip(results['ids'][0], results['documents'][0], results['distances'][0]), 1
        ):
            # Chroma 返回的是距离，值越小越相似
            # 转换为相似度（简单转换，仅供参考）
            similarity = 1 - distance
            
            print(f"  [{i}] ID: {doc_id}")
            print(f"      相似度: {similarity:.4f}")
            print(f"      内容: {document}")
        
        print()
    
    print("=" * 60)
    print("✅ 检索完成")
    print("=" * 60)


def demonstrate_retrieval_params():
    """演示不同的检索参数"""
    
    print("\n" + "=" * 60)
    print("⚙️  检索参数演示")
    print("=" * 60)
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="knowledge_base")
    
    query = "什么是 AI？"
    
    # 测试不同的 top_k
    print(f"\n查询：{query}")
    print("\n【测试不同的 top_k 值】")
    
    for top_k in [1, 3, 5]:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        print(f"\n  top_k = {top_k}:")
        for doc, dist in zip(results['documents'][0], results['distances'][0]):
            print(f"    - {doc[:40]}... (距离: {dist:.4f})")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    basic_retrieval()
    demonstrate_retrieval_params()
