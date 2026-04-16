# -*- coding: utf-8 -*-
"""
向量相似度搜索实战 - 使用 Chroma 最新 API
安装依赖：pip install chromadb
"""

import chromadb
import os


def main():
    print("=" * 60)
    print("🔍 向量相似度搜索实战")
    print("=" * 60)
    
    # 使用 PersistentClient 进行持久化存储
    persist_dir = "./chroma_db"
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    
    client = chromadb.PersistentClient(path=persist_dir)
    
    # 创建集合
    collection_name = "knowledge_base"
    
    # 如果集合已存在，先删除
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    collection = client.get_or_create_collection(name=collection_name)
    
    # 准备知识库文档
    documents = [
        "RAG（检索增强生成）是一种让大模型通过检索外部知识来增强回答的技术。",
        "向量数据库是存储和检索向量的专用数据库，支持相似度搜索。",
        "Embedding 是将文本转换为向量的技术，是 RAG 的基础。",
        "Chunk 是将长文档切分成小块的策略，影响检索质量。",
        "Rerank 是对检索结果进行二次排序的技术，提高相关性。",
        "Prompt 设计是将检索结果喂给大模型的技巧。",
        "Python 是一门流行的编程语言，广泛用于 AI 开发。",
        "FastAPI 是一个高性能的 Python Web 框架。"
    ]
    
    # 添加文档
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    
    print("\n📚 知识库构建完成！")
    print(f"   文档数量：{collection.count()}")
    print("=" * 60)
    
    # 执行检索
    queries = [
        "什么是 RAG？",
        "如何让大模型更聪明？",
        "Python 相关技术有哪些？"
    ]
    
    for query in queries:
        print(f"\n🔍 查询：{query}")
        print("-" * 60)
        
        # 执行相似度搜索
        results = collection.query(
            query_texts=[query],
            n_results=3  # 返回最相似的 3 个文档
        )
        
        # 打印结果
        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            similarity = 1 - distance  # Chroma 使用距离，转换为相似度
            print(f"  [{i+1}] 相似度: {similarity:.4f}")
            print(f"      {doc}")
    
    print("\n" + "=" * 60)
    print("✅ 搜索演示完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
