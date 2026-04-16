# -*- coding: utf-8 -*-
"""
使用 Cross-Encoder 进行精准 Rerank
演示 Cross-Encoder 的使用方法和效果

安装依赖：pip install sentence-transformers
"""

from sentence_transformers import CrossEncoder
from typing import List, Tuple, Optional


def cross_encoder_rerank(
    query: str,
    documents: List[str],
    model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    top_k: int = 3,
    show_progress: bool = False
) -> List[Tuple[str, float]]:
    """
    使用 Cross-Encoder 进行重排序
    
    原理：
    1. 将 Query 和每个 Doc 组成 pair: [CLS] Query [SEP] Doc [SEP]
    2. 输入 Cross-Encoder 模型
    3. 模型输出每个 pair 的相关性分数（深度交互）
    4. 按分数排序返回
    
    Args:
        query: 查询文本
        documents: 文档列表
        model_name: Cross-Encoder 模型名称
        top_k: 返回的文档数量
        show_progress: 是否显示进度条
    
    Returns:
        results: [(文档, 分数), ...]
    """
    # 加载模型
    if show_progress:
        print(f"🔄 加载 Cross-Encoder 模型: {model_name}")
    
    model = CrossEncoder(model_name)
    
    # 构建 Query-Doc pairs
    pairs = [[query, doc] for doc in documents]
    
    # 批量预测分数
    scores = model.predict(pairs, show_progress_bar=show_progress)
    
    # 组合结果并排序
    results = list(zip(documents, scores))
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:top_k]


def demo_cross_encoder():
    """演示 Cross-Encoder Rerank"""
    
    print("=" * 60)
    print("🎯 Cross-Encoder Rerank 演示")
    print("=" * 60)
    
    query = "如何安装 Python"
    
    documents = [
        "Python 是一门流行的编程语言，语法简洁优雅。",
        "Python 安装教程：访问官网 python.org 下载对应系统的安装包。",
        "机器学习是 AI 的一个分支，使用算法从数据中学习。",
        "Python 环境配置：安装完成后，需要配置环境变量。",
        "FastAPI 是一个高性能的 Python Web 框架。",
        "深度学习使用多层神经网络进行特征学习。",
        "Python pip 使用：使用 pip 安装第三方库的方法。",
        "自然语言处理（NLP）让计算机能够理解人类语言。"
    ]
    
    print(f"\n查询：{query}")
    print(f"文档数量：{len(documents)}")
    
    # 显示原始文档
    print("\n【原始文档】")
    print("-" * 60)
    for i, doc in enumerate(documents, 1):
        print(f"  [{i}] {doc}")
    
    # 执行 Rerank
    print("\n" + "=" * 60)
    print("🔄 执行 Cross-Encoder Rerank...")
    print("=" * 60)
    
    reranked = cross_encoder_rerank(query, documents, top_k=5, show_progress=True)
    
    print("\n【Rerank 后的 Top-5】")
    print("-" * 60)
    
    for i, (doc, score) in enumerate(reranked, 1):
        print(f"  [{i}] 分数: {score:.4f}")
        print(f"      {doc}\n")
    
    print("=" * 60)
    print("💡 观察：与'安装 Python'最相关的文档排在最前面")
    print("=" * 60)


def compare_rerank_methods():
    """对比不同 Rerank 方法的效果"""
    
    print("\n" + "=" * 60)
    print("📊 Rerank 方法对比")
    print("=" * 60)
    
    query = "RAG 技术介绍"
    
    documents = [
        "RAG 是检索增强生成技术的缩写。",
        "机器学习是人工智能的一个分支。",
        "RAG 技术结合了检索和生成两个阶段。",
        "Python 是一门编程语言。",
        "RAG 在问答系统中应用广泛。",
        "深度学习使用神经网络进行学习。",
        "RAG 系统需要向量数据库支持。"
    ]
    
    print(f"\n查询：{query}\n")
    
    # 方法1：简单关键词匹配
    print("【方法1：简单关键词匹配】")
    print("-" * 60)
    
    def simple_keyword_match(query, docs, top_k):
        query_chars = set(query.lower())
        scored = []
        for doc in docs:
            doc_chars = set(doc.lower())
            overlap = len(query_chars & doc_chars)
            scored.append((doc, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    simple_results = simple_keyword_match(query, documents, 3)
    
    for i, (doc, score) in enumerate(simple_results, 1):
        print(f"  [{i}] 分数: {score} - {doc[:40]}")
    
    # 方法2：Cross-Encoder
    print("\n【方法2：Cross-Encoder】")
    print("-" * 60)
    
    print("  加载模型...")
    reranked = cross_encoder_rerank(query, documents, top_k=3, show_progress=False)
    
    for i, (doc, score) in enumerate(reranked, 1):
        print(f"  [{i}] 分数: {score:.4f} - {doc[:40]}")
    
    print("\n" + "=" * 60)
    print("💡 结论：")
    print("   - 关键词匹配：简单快速，但语义理解有限")
    print("   - Cross-Encoder：语义理解强，排序更精准")
    print("=" * 60)


def explain_cross_encoder_vs_bi_encoder():
    """解释 Cross-Encoder 与 Bi-Encoder 的区别"""
    
    print("\n" + "=" * 60)
    print("📚 Cross-Encoder vs Bi-Encoder")
    print("=" * 60)
    
    print("""
┌─────────────────────────────────────────────────────────────┐
│                    Bi-Encoder（双塔模型）                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Query → [Encoder] → Query向量 ─┐                    │  │
│  │                                  ├→ 相似度 → 分数    │  │
│  │  Doc   → [Encoder] → Doc向量   ─┘                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  特点：                                                     │
│  ✅ 速度快（Doc 可预先编码）                                │
│  ❌ 精度低（无法捕捉 Query-Doc 深层交互）                   │
│  🎯 适用：召回阶段                                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Cross-Encoder（交叉编码器）                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  [CLS] Query [SEP] Doc [SEP]                         │  │
│  │           ↓                                          │  │
│  │       [Encoder] → 相关性分数                         │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  特点：                                                     │
│  ✅ 精度最高（能捕捉 Query-Doc 深层交互）                   │
│  ❌ 速度慢（每次都要一起计算）                              │
│  🎯 适用：精排阶段（Rerank）                                │
└─────────────────────────────────────────────────────────────┘
    """)
    
    print("\n" + "=" * 60)
    print("✅ 生产环境推荐：Bi-Encoder 召回 + Cross-Encoder 精排")
    print("=" * 60)


def test_different_models():
    """测试不同的 Cross-Encoder 模型"""
    
    print("\n" + "=" * 60)
    print("🧪 不同 Cross-Encoder 模型测试")
    print("=" * 60)
    
    query = "机器学习算法"
    
    documents = [
        "机器学习算法原理与应用。",
        "Python 编程语言介绍。",
        "深度学习神经网络详解。",
        "机器学习算法优化方法。",
        "自然语言处理技术。"
    ]
    
    # 测试不同模型
    models = [
        ('cross-encoder/ms-marco-TinyBERT-L-2', 'TinyBERT（小模型，速度快）'),
        ('cross-encoder/ms-marco-MiniLM-L-6-v2', 'MiniLM（中等模型，平衡）'),
        # ('cross-encoder/ms-marco-MiniLM-L-12-v2', 'MiniLM-L-12（大模型，精度高）'),
    ]
    
    print(f"\n查询：{query}\n")
    
    for model_name, model_desc in models:
        print(f"\n【模型：{model_desc}】")
        print(f"模型名称：{model_name}")
        print("-" * 60)
        
        try:
            reranked = cross_encoder_rerank(query, documents, model_name, top_k=3)
            
            for i, (doc, score) in enumerate(reranked, 1):
                print(f"  [{i}] 分数: {score:.4f} - {doc[:40]}")
        
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
    
    print("\n" + "=" * 60)
    print("💡 提示：")
    print("   - TinyBERT：速度最快，适合实时场景")
    print("   - MiniLM-L-6：平衡速度和精度，推荐使用")
    print("   - MiniLM-L-12：精度最高，但速度较慢")
    print("=" * 60)


if __name__ == "__main__":
    demo_cross_encoder()
    compare_rerank_methods()
    explain_cross_encoder_vs_bi_encoder()
    test_different_models()
