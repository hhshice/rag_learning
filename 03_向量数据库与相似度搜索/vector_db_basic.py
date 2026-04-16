# -*- coding: utf-8 -*-
"""
向量数据库基础示例 - 使用 Chroma 最新 API
安装依赖：pip install chromadb

注意：Chroma 最新版本已简化 API，不再需要 Settings 和 chroma_db_impl 等配置
"""

import chromadb
from chromadb.config import Settings
import os


def main():
    print("=" * 60)
    print("📚 向量数据库基础示例（Chroma 最新 API）")
    print("=" * 60)
    
    # 方式 1：内存模式（数据在程序结束后丢失）
    # chroma_client = chromadb.Client()
    
    # 方式 2：持久化模式（推荐，数据保存到磁盘）
    persist_dir = "./chroma_db"
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    
    # 使用 PersistentClient 进行持久化存储
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    
    print(f"✅ Chroma 客户端已初始化")
    print(f"   数据保存位置：{persist_dir}")
    
    # 创建或获取集合（相当于"表"）
    collection_name = "documents"
    
    # 使用 get_or_create_collection 避免重复创建报错
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "我的第一个向量数据库"}
    )
    
    print(f"✅ 集合已就绪：{collection_name}")
    print(f"   当前文档数量：{collection.count()}")
    
    # 准备文档
    documents = [
        "RAG 是检索增强生成技术，让大模型能基于外部知识回答问题。",
        "向量数据库用于存储和检索文本的向量表示，是 RAG 的核心组件。",
        "Python 是一门流行的编程语言，广泛用于数据科学和 AI 开发。",
        "深度学习是机器学习的一个分支，使用神经网络进行特征学习。",
        "FastAPI 是一个现代的 Python Web 框架，用于构建 API。"
    ]
    
    # 生成唯一 ID
    doc_ids = [f"doc_{i}" for i in range(len(documents))]
    
    # 使用 upsert 添加文档（避免重复插入）
    # Chroma 会自动使用默认的 Embedding 模型进行向量化
    collection.upsert(
        ids=doc_ids,
        documents=documents
    )
    
    print(f"\n✅ 文档已添加到向量数据库")
    print(f"   当前文档数量：{collection.count()}")
    
    # 演示：查询一条数据
    print("\n" + "=" * 60)
    print("🔍 演示查询")
    print("=" * 60)
    
    query = "RAG 是什么？"
    
    # 执行相似度搜索
    # query_texts: Chroma 会自动将查询文本转换为向量
    # n_results: 返回最相似的 n 个结果
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    print(f"查询：{query}")
    print(f"\n最相似的文档：")
    
    # 解析查询结果
    for i, (doc_id, document, distance) in enumerate(
        zip(results['ids'][0], results['documents'][0], results['distances'][0]), 1
    ):
        # Chroma 返回的是距离（distance），值越小越相似
        # 可以转换为相似度：similarity = 1 - distance（简单转换，仅供参考）
        similarity = 1 - distance
        print(f"\n  [{i}] ID: {doc_id}")
        print(f"      相似度: {similarity:.4f}")
        print(f"      内容: {document}")
    
    # 演示：获取集合信息
    print("\n" + "=" * 60)
    print("📊 集合信息")
    print("=" * 60)
    print(f"集合名称：{collection.name}")
    print(f"文档数量：{collection.count()}")
    print(f"元数据：{collection.metadata}")
    
    # 演示：查看所有集合
    print("\n" + "=" * 60)
    print("📂 所有集合")
    print("=" * 60)
    all_collections = chroma_client.list_collections()
    for coll in all_collections:
        print(f"  - {coll.name} (文档数: {coll.count()})")
    
    print("\n" + "=" * 60)
    print("✅ 示例运行完成！")
    print("=" * 60)
    print("\n💡 提示：")
    print("   - 使用 chromadb.Client() 创建内存客户端")
    print("   - 使用 chromadb.PersistentClient(path='...') 创建持久化客户端")
    print("   - 使用 upsert() 而非 add() 可避免重复插入")


if __name__ == "__main__":
    main()
