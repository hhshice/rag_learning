# -*- coding: utf-8 -*-
"""
高级检索示例
演示元数据过滤、相似度阈值、多条件组合检索

安装依赖：pip install chromadb
"""

import chromadb
from typing import List, Dict, Any, Optional


def create_sample_collection():
    """创建示例数据集合"""
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="advanced_retrieval_demo",
        metadata={"description": "高级检索演示数据集"}
    )
    
    # 准备带元数据的文档
    documents = [
        {
            "id": "doc_1",
            "content": "Python 是一门流行的编程语言，语法简洁，适合初学者入门学习。",
            "metadata": {"category": "编程", "level": "入门", "author": "张三", "year": 2024}
        },
        {
            "id": "doc_2",
            "content": "Python 高级特性：装饰器、生成器、上下文管理器等进阶技巧详解。",
            "metadata": {"category": "编程", "level": "高级", "author": "李四", "year": 2024}
        },
        {
            "id": "doc_3",
            "content": "RAG 入门教程：检索增强生成技术基础概念与实践应用。",
            "metadata": {"category": "AI", "level": "入门", "author": "张三", "year": 2025}
        },
        {
            "id": "doc_4",
            "content": "RAG 进阶优化：向量数据库选型、检索策略调优、性能提升技巧。",
            "metadata": {"category": "AI", "level": "高级", "author": "王五", "year": 2025}
        },
        {
            "id": "doc_5",
            "content": "FastAPI 快速入门：构建第一个 RESTful API 服务。",
            "metadata": {"category": "Web开发", "level": "入门", "author": "张三", "year": 2024}
        },
        {
            "id": "doc_6",
            "content": "FastAPI 高级特性：依赖注入、中间件、异步编程实战。",
            "metadata": {"category": "Web开发", "level": "高级", "author": "李四", "year": 2025}
        },
        {
            "id": "doc_7",
            "content": "机器学习入门：监督学习、非监督学习基础概念介绍。",
            "metadata": {"category": "AI", "level": "入门", "author": "王五", "year": 2023}
        },
        {
            "id": "doc_8",
            "content": "深度学习实战：神经网络模型训练与调优方法。",
            "metadata": {"category": "AI", "level": "高级", "author": "王五", "year": 2024}
        }
    ]
    
    # 如果集合为空，添加文档
    if collection.count() == 0:
        collection.upsert(
            ids=[doc["id"] for doc in documents],
            documents=[doc["content"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents]
        )
        print(f"✅ 已添加 {len(documents)} 个文档到集合")
    else:
        print(f"ℹ️  集合已有 {collection.count()} 个文档")
    
    return collection


def demo_metadata_filtering():
    """演示元数据过滤"""
    
    print("=" * 60)
    print("🏷️  元数据过滤演示")
    print("=" * 60)
    
    collection = create_sample_collection()
    
    query = "入门教程"
    
    # 1. 单条件过滤
    print(f"\n【查询】：{query}")
    print("\n【过滤条件】：level = '入门'")
    print("-" * 60)
    
    results = collection.query(
        query_texts=[query],
        n_results=5,
        where={"level": "入门"}
    )
    
    for i, (doc, meta, dist) in enumerate(
        zip(results['documents'][0], results['metadatas'][0], results['distances'][0]), 1
    ):
        similarity = 1 - dist
        print(f"  [{i}] [{meta['category']}] 相似度: {similarity:.4f}")
        print(f"      {doc[:50]}")
    
    # 2. 多条件过滤（AND）
    print(f"\n【过滤条件】：level='入门' AND category='AI'")
    print("-" * 60)
    
    results = collection.query(
        query_texts=[query],
        n_results=5,
        where={
            "$and": [
                {"level": "入门"},
                {"category": "AI"}
            ]
        }
    )
    
    for i, (doc, meta, dist) in enumerate(
        zip(results['documents'][0], results['metadatas'][0], results['distances'][0]), 1
    ):
        similarity = 1 - dist
        print(f"  [{i}] [{meta['category']}] 相似度: {similarity:.4f}")
        print(f"      {doc[:50]}")
    
    # 3. 多条件过滤（OR）
    print(f"\n【过滤条件】：category='编程' OR category='Web开发'")
    print("-" * 60)
    
    results = collection.query(
        query_texts=[query],
        n_results=5,
        where={
            "$or": [
                {"category": "编程"},
                {"category": "Web开发"}
            ]
        }
    )
    
    for i, (doc, meta, dist) in enumerate(
        zip(results['documents'][0], results['metadatas'][0], results['distances'][0]), 1
    ):
        similarity = 1 - dist
        print(f"  [{i}] [{meta['category']}] 相似度: {similarity:.4f}")
        print(f"      {doc[:50]}")
    
    # 4. 范围过滤
    print(f"\n【过滤条件】：year >= 2024")
    print("-" * 60)
    
    results = collection.query(
        query_texts=[query],
        n_results=5,
        where={"year": {"$gte": 2024}}
    )
    
    for i, (doc, meta, dist) in enumerate(
        zip(results['documents'][0], results['metadatas'][0], results['distances'][0]), 1
    ):
        similarity = 1 - dist
        print(f"  [{i}] [{meta['year']}] 相似度: {similarity:.4f}")
        print(f"      {doc[:50]}")
    
    print("\n" + "=" * 60)


def demo_threshold_filtering():
    """演示相似度阈值过滤"""
    
    print("\n" + "=" * 60)
    print("🎯 相似度阈值过滤演示")
    print("=" * 60)
    
    collection = create_sample_collection()
    
    query = "深度学习神经网络"
    
    print(f"\n【查询】：{query}")
    
    # 先检索更多结果
    results = collection.query(
        query_texts=[query],
        n_results=8
    )
    
    # 测试不同阈值
    thresholds = [0.4, 0.6, 0.8]
    
    for threshold in thresholds:
        print(f"\n【阈值 = {threshold}】")
        print("-" * 60)
        
        filtered_count = 0
        
        for doc, dist, meta in zip(
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0]
        ):
            similarity = 1 - dist
            
            if similarity >= threshold:
                print(f"  ✅ [{similarity:.4f}] {doc[:50]}")
                filtered_count += 1
            else:
                print(f"  ❌ [{similarity:.4f}] {doc[:50]} (被过滤)")
        
        print(f"\n  通过阈值的结果数：{filtered_count} / {len(results['documents'][0])}")
    
    print("\n" + "=" * 60)
    print("💡 提示：")
    print("   - 阈值太高：可能漏掉相关结果")
    print("   - 阈值太低：可能包含不相关结果")
    print("   - 推荐范围：0.6-0.8")
    print("=" * 60)


def retrieve_with_strategy(
    collection,
    query: str,
    filters: Optional[Dict] = None,
    threshold: float = 0.6,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    组合检索策略
    
    流程：
    1. 元数据过滤缩小范围
    2. 向量检索找相似文档
    3. 相似度阈值过滤低质量结果
    
    Args:
        collection: Chroma 集合对象
        query: 查询文本
        filters: 元数据过滤条件
        threshold: 相似度阈值
        top_k: 返回结果数量
    
    Returns:
        results: 检索结果列表
    """
    # 步骤1: 检索（带过滤）
    raw_results = collection.query(
        query_texts=[query],
        n_results=top_k * 2,  # 多召回一些
        where=filters
    )
    
    # 步骤2: 阈值过滤
    filtered_results = []
    
    for doc_id, doc, dist, meta in zip(
        raw_results['ids'][0],
        raw_results['documents'][0],
        raw_results['distances'][0],
        raw_results['metadatas'][0]
    ):
        similarity = 1 - dist
        
        if similarity >= threshold:
            filtered_results.append({
                "id": doc_id,
                "content": doc,
                "similarity": similarity,
                "metadata": meta
            })
    
    # 步骤3: 返回 top-k
    return filtered_results[:top_k]


def demo_combined_strategy():
    """演示组合检索策略"""
    
    print("\n" + "=" * 60)
    print("🔧 组合检索策略演示")
    print("=" * 60)
    
    collection = create_sample_collection()
    
    # 场景1：查找 AI 领域的入门教程
    print("\n【场景1】：查找 AI 领域的入门教程")
    print("-" * 60)
    
    query = "AI 入门教程"
    filters = {
        "$and": [
            {"category": "AI"},
            {"level": "入门"}
        ]
    }
    
    results = retrieve_with_strategy(
        collection, query,
        filters=filters,
        threshold=0.5,
        top_k=3
    )
    
    print(f"查询：{query}")
    print(f"过滤：category='AI' AND level='入门'")
    print(f"阈值：0.5\n")
    
    for i, result in enumerate(results, 1):
        print(f"  [{i}] 相似度: {result['similarity']:.4f}")
        print(f"      内容: {result['content'][:50]}")
        print(f"      元数据: {result['metadata']}\n")
    
    # 场景2：查找张三编写的 2024 年文档
    print("\n【场景2】：查找张三编写的 2024 年文档")
    print("-" * 60)
    
    query = "教程文档"
    filters = {
        "$and": [
            {"author": "张三"},
            {"year": {"$gte": 2024}}
        ]
    }
    
    results = retrieve_with_strategy(
        collection, query,
        filters=filters,
        threshold=0.4,
        top_k=3
    )
    
    print(f"查询：{query}")
    print(f"过滤：author='张三' AND year>=2024")
    print(f"阈值：0.4\n")
    
    for i, result in enumerate(results, 1):
        print(f"  [{i}] 相似度: {result['similarity']:.4f}")
        print(f"      内容: {result['content'][:50]}")
        print(f"      元数据: {result['metadata']}\n")
    
    print("=" * 60)


def demo_where_operators():
    """演示 Chroma 的 where 过滤操作符"""
    
    print("\n" + "=" * 60)
    print("📚 Chroma Where 操作符演示")
    print("=" * 60)
    
    collection = create_sample_collection()
    
    operators = [
        {
            "name": "$eq (等于)",
            "filter": {"level": "入门"},
            "description": "level = '入门'"
        },
        {
            "name": "$ne (不等于)",
            "filter": {"level": {"$ne": "入门"}},
            "description": "level != '入门'"
        },
        {
            "name": "$gte (大于等于)",
            "filter": {"year": {"$gte": 2024}},
            "description": "year >= 2024"
        },
        {
            "name": "$lte (小于等于)",
            "filter": {"year": {"$lte": 2024}},
            "description": "year <= 2024"
        },
        {
            "name": "$in (包含)",
            "filter": {"category": {"$in": ["AI", "编程"]}},
            "description": "category IN ['AI', '编程']"
        },
        {
            "name": "$nin (不包含)",
            "filter": {"category": {"$nin": ["AI"]}},
            "description": "category NOT IN ['AI']"
        }
    ]
    
    query = "教程"
    
    for op in operators:
        print(f"\n【{op['name']}】")
        print(f"描述：{op['description']}")
        print("-" * 60)
        
        results = collection.query(
            query_texts=[query],
            n_results=3,
            where=op['filter']
        )
        
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            print(f"  - [{meta['category']}] {doc[:40]}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_metadata_filtering()
    demo_threshold_filtering()
    demo_combined_strategy()
    demo_where_operators()
