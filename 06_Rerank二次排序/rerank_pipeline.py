# -*- coding: utf-8 -*-
"""
完整的检索 + Rerank 流程
演示从向量数据库检索到重排序的完整流程

安装依赖：pip install chromadb sentence-transformers
"""

import chromadb
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any


def create_demo_collection():
    """创建演示用的集合"""
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="rerank_pipeline_demo",
        metadata={"description": "Rerank 流程演示数据集"}
    )
    
    # 准备文档（模拟知识库）
    documents = [
        "RAG（检索增强生成）是一种让大模型通过检索外部知识来增强回答的技术。",
        "向量数据库用于存储和检索文本的向量表示，支持高效的相似度搜索。",
        "Embedding 是将文本转换为向量的技术，是 RAG 系统的基础组件。",
        "Python 是一门流行的编程语言，广泛用于数据科学和 AI 开发。",
        "FastAPI 是一个现代的 Python Web 框架，用于构建高性能 API。",
        "机器学习是人工智能的一个分支，使用算法从数据中学习模式。",
        "深度学习使用多层神经网络进行特征学习，是机器学习的子领域。",
        "自然语言处理（NLP）让计算机能够理解和生成人类语言。",
        "RAG 的核心流程包括：文档加载、切分、向量化、检索、生成。",
        "Cross-Encoder 是一种用于重排序的模型，能够捕捉 Query-Doc 的深层交互。",
        "向量检索速度快但精度有限，Rerank 可以显著提升检索质量。",
        "RAG 适用于企业知识库、智能客服、文档问答等场景。",
        "文档切分是 RAG 的关键步骤，直接影响检索效果。",
        "Prompt 设计是将检索结果喂给大模型的重要技巧。",
        "RAG 系统优化包括召回率、精度、速度等多个维度。",
        "向量相似度使用余弦相似度或欧氏距离计算。",
        "Rerank 的作用是对检索结果进行二次排序，提升相关性。",
        "知识库构建需要考虑数据质量、更新频率、检索效率等因素。",
        "大模型在 RAG 中负责理解问题和生成回答。",
        "检索策略包括向量检索、关键词检索、混合检索等。"
    ]
    
    if collection.count() == 0:
        collection.upsert(
            ids=[f"doc_{i}" for i in range(len(documents))],
            documents=documents
        )
        print(f"✅ 已添加 {len(documents)} 个文档到知识库")
    else:
        print(f"ℹ️  知识库已有 {collection.count()} 个文档")
    
    return collection


def retrieve_and_rerank(
    collection,
    query: str,
    retrieve_k: int = 20,
    rerank_k: int = 5,
    rerank_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    rerank_threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    完整的检索 + Rerank 流程
    
    流程：
    1. 从向量数据库召回 retrieve_k 个文档（粗排）
    2. 使用 Cross-Encoder 对召回文档重排序（精排）
    3. 可选：根据阈值过滤低相关性文档
    4. 返回 top rerank_k 个文档
    
    Args:
        collection: Chroma 集合对象
        query: 查询文本
        retrieve_k: 召回文档数量
        rerank_k: 最终返回文档数量
        rerank_model: Rerank 模型名称
        rerank_threshold: Rerank 分数阈值
    
    Returns:
        final_results: 最终结果列表
    """
    print("=" * 60)
    print("🔍 检索 + Rerank 完整流程")
    print("=" * 60)
    
    # ========== 阶段1：召回（粗排） ==========
    print(f"\n【阶段1：召回】召回 {retrieve_k} 个文档...")
    
    retrieve_results = collection.query(
        query_texts=[query],
        n_results=min(retrieve_k, collection.count())
    )
    
    documents = retrieve_results['documents'][0]
    ids = retrieve_results['ids'][0]
    distances = retrieve_results['distances'][0]
    
    print(f"✅ 召回完成：{len(documents)} 个文档")
    
    # 显示召回的前5个
    print("\n召回的前5个文档（按向量相似度）：")
    print("-" * 60)
    for i, (doc, dist) in enumerate(zip(documents[:5], distances[:5]), 1):
        similarity = 1 - dist
        print(f"  [{i}] 相似度: {similarity:.4f}")
        print(f"      {doc[:60]}")
    
    # ========== 阶段2：Rerank（精排） ==========
    print(f"\n【阶段2：Rerank】使用 Cross-Encoder 重排序...")
    print(f"  模型: {rerank_model}")
    
    # 加载模型
    model = CrossEncoder(rerank_model)
    
    # 构建 pairs 并预测
    pairs = [[query, doc] for doc in documents]
    scores = model.predict(pairs)
    
    # 组合结果
    reranked_results = []
    for i, (doc_id, doc, score) in enumerate(zip(ids, documents, scores)):
        reranked_results.append({
            "id": doc_id,
            "content": doc,
            "rerank_score": float(score),
            "vector_similarity": 1 - distances[i],
            "original_rank": i + 1
        })
    
    # 按 rerank 分数排序
    reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
    
    print(f"✅ Rerank 完成")
    
    # 阈值过滤
    if rerank_threshold > 0:
        reranked_results = [
            r for r in reranked_results
            if r['rerank_score'] >= rerank_threshold
        ]
        print(f"✅ 阈值过滤（>= {rerank_threshold}）：保留 {len(reranked_results)} 个文档")
    
    # 返回 top-k
    final_results = reranked_results[:rerank_k]
    
    # 显示最终结果
    print(f"\n【最终结果】Top-{len(final_results)}：")
    print("-" * 60)
    
    for i, result in enumerate(final_results, 1):
        print(f"  [{i}] Rerank分数: {result['rerank_score']:.4f}")
        print(f"      向量相似度: {result['vector_similarity']:.4f}")
        print(f"      原始排名: {result['original_rank']}")
        print(f"      内容: {result['content'][:60]}")
        print()
    
    return final_results


def demonstrate_pipeline():
    """演示完整的检索 + Rerank 流程"""
    
    print("\n" + "=" * 60)
    print("🎯 完整流程演示")
    print("=" * 60)
    
    # 创建数据集
    collection = create_demo_collection()
    
    # 测试查询
    queries = [
        "RAG 技术的核心流程是什么",
        "如何优化 RAG 系统的检索效果"
    ]
    
    for query in queries:
        print("\n" + "=" * 60)
        results = retrieve_and_rerank(
            collection,
            query,
            retrieve_k=10,
            rerank_k=3
        )
    
    print("\n" + "=" * 60)
    print("✅ 演示完成")
    print("=" * 60)


def compare_with_without_rerank():
    """对比有无 Rerank 的效果差异"""
    
    print("\n" + "=" * 60)
    print("📊 对比：有无 Rerank")
    print("=" * 60)
    
    collection = create_demo_collection()
    
    query = "RAG 系统如何优化"
    
    # 1. 无 Rerank
    print(f"\n查询：{query}")
    print("\n【无 Rerank】（纯向量检索）")
    print("-" * 60)
    
    results = collection.query(
        query_texts=[query],
        n_results=5
    )
    
    for i, (doc, dist) in enumerate(
        zip(results['documents'][0], results['distances'][0]), 1
    ):
        similarity = 1 - dist
        print(f"  [{i}] 相似度: {similarity:.4f} - {doc[:50]}")
    
    # 2. 有 Rerank
    print("\n【有 Rerank】（向量检索 + Cross-Encoder）")
    print("-" * 60)
    
    final_results = retrieve_and_rerank(
        collection,
        query,
        retrieve_k=10,
        rerank_k=5
    )
    
    print("\n" + "=" * 60)
    print("💡 观察：Rerank 后，与查询最相关的文档排序更精准")
    print("=" * 60)


def analyze_rank_changes():
    """分析 Rerank 前后的排名变化"""
    
    print("\n" + "=" * 60)
    print("📈 排名变化分析")
    print("=" * 60)
    
    collection = create_demo_collection()
    
    query = "Cross-Encoder 的作用是什么"
    
    # 召回
    results = collection.query(
        query_texts=[query],
        n_results=10
    )
    
    documents = results['documents'][0]
    
    # Rerank
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[query, doc] for doc in documents]
    scores = model.predict(pairs)
    
    # 组合并排序
    ranked = [(i, doc, score) for i, (doc, score) in enumerate(zip(documents, scores), 1)]
    ranked.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n查询：{query}")
    print("\n排名变化（原始排名 → Rerank 后排名）：")
    print("-" * 60)
    
    for new_rank, (old_rank, doc, score) in enumerate(ranked[:5], 1):
        change = old_rank - new_rank
        change_str = f"↑{change}" if change > 0 else (f"↓{abs(change)}" if change < 0 else "→")
        
        print(f"  [{new_rank}] 原排名: {old_rank} {change_str}")
        print(f"      Rerank分数: {score:.4f}")
        print(f"      {doc[:50]}\n")
    
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_pipeline()
    compare_with_without_rerank()
    analyze_rank_changes()
