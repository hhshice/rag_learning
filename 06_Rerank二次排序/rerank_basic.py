# -*- coding: utf-8 -*-
"""
基础 Rerank 示例
演示重排序的基本概念

这个示例使用简单的关键词匹配进行重排序，
实际应用中应该使用 Cross-Encoder 等更高级的方法。
"""

from typing import List, Tuple


def simple_rerank(
    query: str,
    documents: List[str],
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    简单的重排序示例（基于关键词匹配）
    
    这只是演示 Rerank 的概念，实际应用中会使用 Cross-Encoder
    
    原理：
    1. 计算查询和每个文档的关键词重叠度
    2. 按重叠度降序排序
    3. 返回 top-k 文档
    
    Args:
        query: 查询文本
        documents: 文档列表
        top_k: 返回的文档数量
    
    Returns:
        reranked_docs: [(文档, 分数), ...]
    """
    # 中文分词（简单实现：按字符拆分）
    def tokenize(text):
        return set(text)
    
    query_chars = tokenize(query.lower())
    
    scored_docs = []
    for doc in documents:
        doc_chars = tokenize(doc.lower())
        # 计算字符重叠度
        overlap = len(query_chars & doc_chars)
        # 归一化分数
        score = overlap / len(query_chars) if query_chars else 0
        scored_docs.append((doc, score))
    
    # 按分数降序排序
    scored_docs.sort(reverse=True, key=lambda x: x[1])
    
    return scored_docs[:top_k]


def demonstrate_rerank_concept():
    """演示 Rerank 的核心概念"""
    
    print("=" * 60)
    print("🎯 Rerank 概念演示")
    print("=" * 60)
    
    # 场景：用户查询 Python 安装教程
    query = "Python 安装教程"
    
    # 检索返回的文档（模拟召回结果）
    documents = [
        "Python 是一门流行的编程语言，语法简洁优雅。",
        "Python 安装教程：访问官网下载对应系统的安装包。",
        "机器学习是人工智能的一个分支，使用算法学习。",
        "Python 环境配置：安装后需配置环境变量。",
        "FastAPI 是一个高性能的 Python Web 框架。",
        "深度学习使用多层神经网络进行特征学习。",
        "Python pip 使用：pip install 安装第三方库。",
        "自然语言处理让计算机理解人类语言。"
    ]
    
    print(f"\n查询：{query}")
    print(f"召回文档数量：{len(documents)}")
    
    # 显示召回结果（原始顺序）
    print("\n【召回结果】（按向量相似度排序）")
    print("-" * 60)
    
    # 模拟向量检索的排序（这里用索引代替）
    for i, doc in enumerate(documents, 1):
        print(f"  [{i}] {doc}")
    
    # 执行 Rerank
    print("\n" + "=" * 60)
    print("【Rerank 后的结果】")
    print("-" * 60)
    
    reranked = simple_rerank(query, documents, top_k=3)
    
    for i, (doc, score) in enumerate(reranked, 1):
        print(f"  [{i}] 分数: {score:.4f}")
        print(f"      {doc}\n")
    
    print("=" * 60)


def compare_before_after_rerank():
    """对比 Rerank 前后的差异"""
    
    print("\n" + "=" * 60)
    print("📊 Rerank 前后对比")
    print("=" * 60)
    
    query = "机器学习算法"
    
    documents = [
        "Python 编程语言介绍。",
        "机器学习算法原理与应用。",
        "深度学习框架 PyTorch。",
        "机器学习算法分类：监督学习、非监督学习。",
        "自然语言处理技术。",
        "机器学习算法优化方法。"
    ]
    
    print(f"\n查询：{query}\n")
    
    # Rerank 前
    print("【Rerank 前】（原始顺序）")
    print("-" * 60)
    for i, doc in enumerate(documents, 1):
        print(f"  [{i}] {doc}")
    
    # Rerank 后
    print("\n【Rerank 后】")
    print("-" * 60)
    
    reranked = simple_rerank(query, documents, top_k=3)
    
    for i, (doc, score) in enumerate(reranked, 1):
        print(f"  [{i}] 分数: {score:.4f} - {doc}")
    
    print("\n" + "=" * 60)
    print("💡 观察：Rerank 后，与查询最相关的文档排在前面")
    print("=" * 60)


def explain_two_stage_retrieval():
    """解释两阶段检索架构"""
    
    print("\n" + "=" * 60)
    print("📚 两阶段检索架构")
    print("=" * 60)
    
    print("""
┌─────────────────────────────────────────────────────────┐
│                    用户查询                              │
│                       ↓                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │     第一阶段：召回（Recall）                      │   │
│  │  ┌──────────────────────────────────────────┐  │   │
│  │  │ 目标：尽可能多地召回相关文档              │  │   │
│  │  │ 方法：向量检索、关键词检索、混合检索      │  │   │
│  │  │ 数量：20-100 个                          │  │   │
│  │  │ 特点：速度快，但精度有限                  │  │   │
│  │  └──────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                       ↓                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │     第二阶段：精排（Rerank）                      │   │
│  │  ┌──────────────────────────────────────────┐  │   │
│  │  │ 目标：对召回的文档精确排序                │  │   │
│  │  │ 方法：Cross-Encoder、LLM Rerank          │  │   │
│  │  │ 数量：3-10 个                            │  │   │
│  │  │ 特点：精度高，但计算成本较高              │  │   │
│  │  └──────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                       ↓                                  │
│                  最终结果                                │
└─────────────────────────────────────────────────────────┘
    """)
    
    print("\n" + "=" * 60)
    print("✅ 两阶段检索平衡了速度和精度")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_rerank_concept()
    compare_before_after_rerank()
    explain_two_stage_retrieval()
