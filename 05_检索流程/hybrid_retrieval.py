# -*- coding: utf-8 -*-
"""
混合检索示例
演示向量检索 + 关键词检索的组合使用

安装依赖：pip install chromadb
"""

import chromadb
from collections import Counter
import re
from typing import List, Tuple, Dict


def tokenize(text: str) -> List[str]:
    """
    简单分词器
    支持中英文混合文本
    
    Args:
        text: 输入文本
    
    Returns:
        tokens: 词元列表
    """
    # 匹配中文字符和英文单词
    tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text.lower())
    return tokens


def simple_keyword_search(
    documents: List[str],
    query: str,
    top_k: int = 3
) -> List[Tuple[int, float]]:
    """
    简单的关键词检索（BM25 简化版）
    
    原理：
    1. 对查询和文档进行分词
    2. 计算词频和重叠度
    3. 根据评分排序
    
    Args:
        documents: 文档列表
        query: 查询文本
        top_k: 返回结果数量
    
    Returns:
        results: [(文档索引, 分数), ...]
    """
    query_tokens = tokenize(query)
    query_token_set = set(query_tokens)
    
    scores = []
    
    for i, doc in enumerate(documents):
        doc_tokens = tokenize(doc)
        doc_token_set = set(doc_tokens)
        
        # 计算重叠词数量
        overlap = len(query_token_set & doc_token_set)
        
        # 计算词频（TF）
        doc_counter = Counter(doc_tokens)
        tf = sum(doc_counter[token] for token in query_tokens)
        
        # 计算文档长度归一化
        doc_length = len(doc_tokens)
        length_norm = 1.0 / (1 + doc_length * 0.01)
        
        # 综合分数
        score = (overlap + tf * 0.1) * length_norm
        
        scores.append((i, score))
    
    # 按分数降序排序
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores[:top_k]


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    归一化分数到 [0, 1] 区间
    
    Args:
        scores: 原始分数字典
    
    Returns:
        normalized: 归一化后的分数字典
    """
    if not scores:
        return {}
    
    max_score = max(scores.values())
    min_score = min(scores.values())
    range_score = max_score - min_score
    
    if range_score == 0:
        # 所有分数相同，归一化为 0.5
        return {k: 0.5 for k in scores}
    
    return {k: (v - min_score) / range_score for k, v in scores.items()}


def hybrid_retrieval(
    collection,
    documents: List[str],
    query: str,
    top_k: int = 3,
    alpha: float = 0.5
) -> List[Tuple[str, float]]:
    """
    混合检索：向量检索 + 关键词检索
    
    流程：
    1. 执行向量检索
    2. 执行关键词检索
    3. 归一化分数
    4. 加权融合
    5. 排序返回
    
    Args:
        collection: Chroma 集合对象
        documents: 文档列表
        query: 查询文本
        top_k: 返回结果数量
        alpha: 向量检索权重（0-1），关键词权重为 1-alpha
    
    Returns:
        results: [(文档ID, 混合分数), ...]
    """
    # 1. 向量检索
    vector_results = collection.query(
        query_texts=[query],
        n_results=min(top_k * 2, len(documents))  # 多召回一些
    )
    
    vector_scores = {}
    for doc_id, distance in zip(vector_results['ids'][0], vector_results['distances'][0]):
        # 距离转换为相似度
        vector_scores[doc_id] = 1 - distance
    
    # 2. 关键词检索
    keyword_results = simple_keyword_search(documents, query, top_k=top_k * 2)
    
    keyword_scores = {}
    for idx, score in keyword_results:
        doc_id = f"doc_{idx}"
        keyword_scores[doc_id] = score
    
    # 3. 归一化分数
    vector_scores_norm = normalize_scores(vector_scores)
    keyword_scores_norm = normalize_scores(keyword_scores)
    
    # 4. 融合分数
    all_doc_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
    
    final_scores = {}
    for doc_id in all_doc_ids:
        v_score = vector_scores_norm.get(doc_id, 0)
        k_score = keyword_scores_norm.get(doc_id, 0)
        
        # 加权融合
        final_scores[doc_id] = alpha * v_score + (1 - alpha) * k_score
    
    # 5. 排序并返回
    sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_results[:top_k]


def compare_retrieval_methods():
    """对比不同检索方法的效果"""
    
    print("=" * 60)
    print("🔍 检索方法对比")
    print("=" * 60)
    
    # 准备测试数据
    documents = [
        "Python 是一门流行的编程语言，语法简洁优雅。",
        "Python 的列表（List）是一种有序可变的数据集合。",
        "RAG 是检索增强生成技术，结合了检索和生成。",
        "Python 安装教程：访问官网 python.org 下载安装包。",
        "FastAPI 是一个高性能的 Python Web 框架。",
        "机器学习使用算法从数据中学习模式和规律。",
        "深度学习基于多层神经网络进行特征学习。",
        "自然语言处理（NLP）让计算机理解人类语言。"
    ]
    
    # 初始化向量数据库
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="hybrid_comparison")
    
    # 添加文档
    if collection.count() == 0:
        collection.upsert(
            ids=[f"doc_{i}" for i in range(len(documents))],
            documents=documents
        )
    
    # 测试查询
    test_queries = [
        "Python 怎么安装",
        "什么是 RAG",
        "神经网络是什么"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"查询：{query}")
        print("=" * 60)
        
        # 1. 纯向量检索
        print("\n【纯向量检索】")
        vector_results = collection.query(
            query_texts=[query],
            n_results=3
        )
        
        for i, (doc, dist) in enumerate(
            zip(vector_results['documents'][0], vector_results['distances'][0]), 1
        ):
            similarity = 1 - dist
            print(f"  [{i}] 相似度: {similarity:.4f}")
            print(f"      {doc[:50]}")
        
        # 2. 纯关键词检索
        print("\n【纯关键词检索】")
        keyword_results = simple_keyword_search(documents, query, top_k=3)
        
        for i, (idx, score) in enumerate(keyword_results, 1):
            print(f"  [{i}] 分数: {score:.4f}")
            print(f"      {documents[idx][:50]}")
        
        # 3. 混合检索
        print("\n【混合检索 (alpha=0.5)】")
        hybrid_results = hybrid_retrieval(
            collection, documents, query, top_k=3, alpha=0.5
        )
        
        for i, (doc_id, score) in enumerate(hybrid_results, 1):
            idx = int(doc_id.split('_')[1])
            print(f"  [{i}] 混合分数: {score:.4f}")
            print(f"      {documents[idx][:50]}")
    
    print("\n" + "=" * 60)


def demo_alpha_tuning():
    """演示 alpha 参数调优"""
    
    print("\n" + "=" * 60)
    print("⚙️  Alpha 参数调优演示")
    print("=" * 60)
    
    # 准备数据
    documents = [
        "Python 教程：从入门到精通",
        "Python 安装与配置指南",
        "RAG 技术详解与应用",
        "Python Web 开发实战"
    ]
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="alpha_tuning")
    
    if collection.count() == 0:
        collection.upsert(
            ids=[f"doc_{i}" for i in range(len(documents))],
            documents=documents
        )
    
    query = "Python 安装教程"
    
    print(f"\n查询：{query}")
    print("\n不同 alpha 值的结果：")
    print("-" * 60)
    
    # 测试不同的 alpha 值
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    for alpha in alphas:
        print(f"\nalpha = {alpha} (向量权重: {alpha}, 关键词权重: {1-alpha})")
        
        results = hybrid_retrieval(collection, documents, query, top_k=2, alpha=alpha)
        
        for i, (doc_id, score) in enumerate(results, 1):
            idx = int(doc_id.split('_')[1])
            print(f"  [{i}] 分数: {score:.4f} - {documents[idx]}")
    
    print("\n" + "=" * 60)
    print("💡 提示：")
    print("   - alpha=1.0: 纯向量检索，语义理解强")
    print("   - alpha=0.0: 纯关键词检索，精确匹配强")
    print("   - alpha=0.5: 平衡模式，综合效果好")
    print("=" * 60)


if __name__ == "__main__":
    compare_retrieval_methods()
    demo_alpha_tuning()
