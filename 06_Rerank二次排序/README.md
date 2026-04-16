# ⑥ Rerank（为什么需要二次排序）

## 🧠 概念解释

### 什么是 Rerank？

> **Rerank = 对检索结果进行二次排序，提升相关性**

在检索阶段，我们通常使用向量检索或混合检索召回一批文档。但这些文档的排序不一定最优，需要 Rerank 进行精排。

```
检索阶段（召回）          Rerank 阶段（精排）
     ↓                        ↓
  Top-K 文档              重排序后的 Top-K
  (可能包含噪音)          (相关性更高)
```

### 为什么需要 Rerank？

| 问题 | 检索阶段的局限 | Rerank 的解决方案 |
|------|---------------|------------------|
| **向量检索语义偏差** | 向量相似度 ≠ 文本相关性 | 用更精确的模型重新打分 |
| **召回文档过多** | Top-K 可能包含无关文档 | 过滤低相关性文档 |
| **排序不够精准** | 向量距离不能完美反映相关性 | 用 Cross-Encoder 精确匹配 |
| **业务规则缺失** | 纯语义检索忽略业务因素 | 结合业务规则重排 |

### Rerank 的核心原理

**两阶段检索架构：**

```
第一阶段：召回（Recall）
  - 目标：尽可能多地召回相关文档
  - 方法：向量检索、关键词检索、混合检索
  - 召回数量：通常 20-100 个
  - 特点：速度快，但精度有限

第二阶段：精排（Rerank）
  - 目标：对召回的文档精确排序
  - 方法：Cross-Encoder、LLM Rerank、规则打分
  - 返回数量：通常 3-10 个
  - 特点：精度高，但计算成本较高
```

### Rerank 方法对比

| 方法 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| **Cross-Encoder** | 将 Query 和 Doc 一起输入模型，输出相关性分数 | 精度最高 | 计算成本高 | 高精度需求 |
| **Bi-Encoder + Cross-Encoder** | 先用 Bi-Encoder 召回，再用 Cross-Encoder 重排 | 平衡速度和精度 | 需要两个模型 | 生产环境推荐 |
| **LLM Rerank** | 用大模型（GPT-4）判断文档相关性 | 效果好，可解释性强 | 成本极高，速度慢 | 高价值场景 |
| **Cohere Rerank API** | 使用 Cohere 的 Rerank 服务 | 开箱即用，效果好 | 需要付费 | 快速集成 |
| **规则打分** | 结合业务规则（时间、热度等）调整分数 | 简单可控 | 缺乏语义理解 | 业务规则明确的场景 |

### Cross-Encoder vs Bi-Encoder

| 特性 | Bi-Encoder（双塔模型） | Cross-Encoder（交叉编码器） |
|------|----------------------|---------------------------|
| **架构** | Query 和 Doc 分别编码，最后计算相似度 | Query 和 Doc 一起输入模型，直接输出分数 |
| **速度** | 快（可预先编码 Doc） | 慢（每次都要一起计算） |
| **精度** | 较低（无法捕捉深层交互） | 最高（能捕捉 Query-Doc 深层交互） |
| **适用阶段** | 召回阶段 | 精排阶段 |
| **代表模型** | sentence-transformers | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

**图示：**

```
Bi-Encoder:
  Query → [Encoder] → Query向量 ─┐
                                  ├→ 相似度计算 → 分数
  Doc   → [Encoder] → Doc向量   ─┘

Cross-Encoder:
  [CLS] Query [SEP] Doc [SEP] → [Encoder] → 相关性分数
  (Query 和 Doc 一起输入，深度交互)
```

---

## 📦 示例代码

### 示例 1：基础 Rerank 示例

```python
# rerank_basic.py
# 最基础的 Rerank 示例

def simple_rerank(query, documents, top_k=3):
    """
    简单的重排序示例（基于关键词匹配）
    
    这只是演示 Rerank 的概念，实际应用中会使用 Cross-Encoder
    
    Args:
        query: 查询文本
        documents: 文档列表
        top_k: 返回的文档数量
    
    Returns:
        reranked_docs: 重排序后的文档列表
    """
    # 计算每个文档与查询的关键词重叠度
    query_words = set(query.lower().split())
    
    scored_docs = []
    for doc in documents:
        doc_words = set(doc.lower().split())
        overlap = len(query_words & doc_words)
        scored_docs.append((overlap, doc))
    
    # 按分数降序排序
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    
    # 返回 top-k
    return [doc for score, doc in scored_docs[:top_k]]


if __name__ == "__main__":
    query = "Python 安装教程"
    
    documents = [
        "Python 是一门流行的编程语言。",
        "Python 安装教程：访问官网下载安装包。",
        "机器学习使用算法从数据中学习。",
        "Python 环境配置与安装步骤详解。",
        "FastAPI 是一个 Python Web 框架。"
    ]
    
    print("查询：", query)
    print("\n原始顺序：")
    for i, doc in enumerate(documents, 1):
        print(f"  [{i}] {doc}")
    
    reranked = simple_rerank(query, documents, top_k=3)
    
    print("\n重排序后：")
    for i, doc in enumerate(reranked, 1):
        print(f"  [{i}] {doc}")
```

---

### 示例 2：使用 Cross-Encoder 进行 Rerank

```python
# rerank_cross_encoder.py
# 使用 Cross-Encoder 进行精准 Rerank

# 安装依赖：pip install sentence-transformers

from sentence_transformers import CrossEncoder
from typing import List, Tuple


def cross_encoder_rerank(
    query: str,
    documents: List[str],
    model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    使用 Cross-Encoder 进行重排序
    
    原理：
    1. 将 Query 和每个 Doc 组成 pair
    2. 输入 Cross-Encoder 模型
    3. 模型输出每个 pair 的相关性分数
    4. 按分数排序返回
    
    Args:
        query: 查询文本
        documents: 文档列表
        model_name: Cross-Encoder 模型名称
        top_k: 返回的文档数量
    
    Returns:
        results: [(文档, 分数), ...]
    """
    # 加载模型
    print(f"🔄 加载 Cross-Encoder 模型: {model_name}")
    model = CrossEncoder(model_name)
    
    # 构建 Query-Doc pairs
    pairs = [[query, doc] for doc in documents]
    
    # 批量预测分数
    scores = model.predict(pairs)
    
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
        "Python 是一门流行的编程语言，语法简洁。",
        "Python 安装教程：访问官网 python.org 下载对应系统的安装包。",
        "机器学习是 AI 的一个分支，使用算法从数据中学习。",
        "Python 环境配置：安装完成后，需要配置环境变量。",
        "FastAPI 是一个高性能的 Python Web 框架。",
        "深度学习使用多层神经网络进行特征学习。",
        "Python pip 安装：使用 pip 安装第三方库。"
    ]
    
    print(f"\n查询：{query}")
    print(f"文档数量：{len(documents)}")
    print("\n原始文档：")
    print("-" * 60)
    for i, doc in enumerate(documents, 1):
        print(f"  [{i}] {doc}")
    
    # 执行 Rerank
    print("\n" + "=" * 60)
    print("🔄 执行 Cross-Encoder Rerank...")
    print("=" * 60)
    
    reranked = cross_encoder_rerank(query, documents, top_k=3)
    
    print("\n重排序后的 Top-3：")
    print("-" * 60)
    for i, (doc, score) in enumerate(reranked, 1):
        print(f"  [{i}] 分数: {score:.4f}")
        print(f"      {doc}\n")
    
    print("=" * 60)


def compare_rerank_methods():
    """对比不同 Rerank 方法"""
    
    print("\n" + "=" * 60)
    print("📊 Rerank 方法对比")
    print("=" * 60)
    
    query = "RAG 是什么"
    
    documents = [
        "RAG 是检索增强生成技术的缩写。",
        "机器学习是人工智能的一个分支。",
        "RAG 技术结合了检索和生成两个阶段。",
        "Python 是一门编程语言。",
        "RAG 在问答系统中应用广泛。"
    ]
    
    print(f"\n查询：{query}\n")
    
    # 方法1：简单关键词匹配
    print("【方法1：简单关键词匹配】")
    print("-" * 60)
    
    query_words = set(query.lower().split())
    scored = []
    for doc in documents:
        doc_words = set(doc.lower().split())
        score = len(query_words & doc_words)
        scored.append((doc, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    
    for i, (doc, score) in enumerate(scored[:3], 1):
        print(f"  [{i}] 分数: {score} - {doc[:40]}")
    
    # 方法2：Cross-Encoder
    print("\n【方法2：Cross-Encoder】")
    print("-" * 60)
    
    reranked = cross_encoder_rerank(query, documents, top_k=3)
    
    for i, (doc, score) in enumerate(reranked, 1):
        print(f"  [{i}] 分数: {score:.4f} - {doc[:40]}")
    
    print("\n" + "=" * 60)
    print("💡 Cross-Encoder 能够捕捉深层语义，效果更好")
    print("=" * 60)


if __name__ == "__main__":
    demo_cross_encoder()
    compare_rerank_methods()
```

---

### 示例 3：完整的检索 + Rerank 流程

```python
# rerank_pipeline.py
# 完整流程：检索 → Rerank → 返回结果

# 安装依赖：pip install chromadb sentence-transformers

import chromadb
from sentence_transformers import CrossEncoder


def retrieve_and_rerank(
    collection,
    query: str,
    retrieve_k: int = 20,
    rerank_k: int = 5,
    rerank_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
):
    """
    完整的检索 + Rerank 流程
    
    流程：
    1. 从向量数据库召回 retrieve_k 个文档
    2. 使用 Cross-Encoder 对召回文档重排序
    3. 返回 top rerank_k 个文档
    
    Args:
        collection: Chroma 集合对象
        query: 查询文本
        retrieve_k: 召回文档数量
        rerank_k: 最终返回文档数量
        rerank_model: Rerank 模型名称
    
    Returns:
        final_results: 最终结果列表
    """
    print("=" * 60)
    print("🔍 检索 + Rerank 完整流程")
    print("=" * 60)
    
    # 阶段1：召回
    print(f"\n【阶段1：召回】召回 {retrieve_k} 个文档...")
    
    retrieve_results = collection.query(
        query_texts=[query],
        n_results=retrieve_k
    )
    
    documents = retrieve_results['documents'][0]
    ids = retrieve_results['ids'][0]
    distances = retrieve_results['distances'][0]
    
    print(f"✅ 召回完成：{len(documents)} 个文档")
    
    # 显示召回的前5个
    print("\n召回的前5个文档：")
    for i, (doc, dist) in enumerate(zip(documents[:5], distances[:5]), 1):
        similarity = 1 - dist
        print(f"  [{i}] 相似度: {similarity:.4f} - {doc[:40]}")
    
    # 阶段2：Rerank
    print(f"\n【阶段2：Rerank】使用 Cross-Encoder 重排序...")
    
    # 加载模型
    print(f"  加载模型: {rerank_model}")
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
            "original_index": i
        })
    
    # 按 rerank 分数排序
    reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
    
    # 返回 top-k
    final_results = reranked_results[:rerank_k]
    
    print(f"✅ Rerank 完成")
    
    # 显示 Rerank 后的结果
    print(f"\n最终结果（Top-{rerank_k}）：")
    print("-" * 60)
    
    for i, result in enumerate(final_results, 1):
        print(f"  [{i}] Rerank分数: {result['rerank_score']:.4f}")
        print(f"      原始排名: {result['original_index'] + 1}")
        print(f"      内容: {result['content'][:50]}")
        print()
    
    return final_results


def create_demo_collection():
    """创建演示用的集合"""
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="rerank_demo",
        metadata={"description": "Rerank 演示数据集"}
    )
    
    # 准备文档
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
        "RAG 系统优化包括召回率、精度、速度等多个维度。"
    ]
    
    if collection.count() == 0:
        collection.upsert(
            ids=[f"doc_{i}" for i in range(len(documents))],
            documents=documents
        )
        print(f"✅ 已添加 {len(documents)} 个文档")
    else:
        print(f"ℹ️  集合已有 {collection.count()} 个文档")
    
    return collection


def demo_full_pipeline():
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


if __name__ == "__main__":
    demo_full_pipeline()
```

---

### 示例 4：使用 Cohere Rerank API（可选）

```python
# rerank_cohere.py
# 使用 Cohere Rerank API 进行重排序

# 安装依赖：pip install cohere

import cohere
import os


def cohere_rerank(
    query: str,
    documents: list,
    model: str = "rerank-multilingual-v2.0",
    top_n: int = 3
):
    """
    使用 Cohere Rerank API
    
    优点：
    - 开箱即用，无需加载模型
    - 支持多语言
    - 效果好
    
    缺点：
    - 需要 API Key
    - 需要付费
    
    Args:
        query: 查询文本
        documents: 文档列表
        model: Rerank 模型名称
        top_n: 返回数量
    
    Returns:
        results: 重排序结果
    """
    # 初始化 Cohere 客户端
    api_key = os.environ.get("COHERE_API_KEY")
    
    if not api_key:
        print("⚠️  需要设置 COHERE_API_KEY 环境变量")
        print("   获取 API Key: https://dashboard.cohere.com/")
        return None
    
    co = cohere.Client(api_key)
    
    # 调用 Rerank API
    results = co.rerank(
        query=query,
        documents=documents,
        model=model,
        top_n=top_n
    )
    
    return results


def demo_cohere_rerank():
    """演示 Cohere Rerank"""
    
    print("=" * 60)
    print("🌐 Cohere Rerank API 演示")
    print("=" * 60)
    
    query = "如何学习 Python"
    
    documents = [
        "Python 是一门流行的编程语言。",
        "Python 安装教程与配置指南。",
        "机器学习算法原理介绍。",
        "Python 学习路线：从入门到精通。",
        "深度学习框架 PyTorch 教程。"
    ]
    
    print(f"\n查询：{query}")
    print(f"文档数量：{len(documents)}")
    
    # 调用 Cohere Rerank
    results = cohere_rerank(query, documents, top_n=3)
    
    if results:
        print("\n重排序结果：")
        print("-" * 60)
        
        for i, result in enumerate(results.results, 1):
            print(f"  [{i}] 相关性分数: {result.relevance_score:.4f}")
            print(f"      文档: {documents[result.index]}")
            print(f"      原始索引: {result.index}\n")
    
    print("=" * 60)
    print("💡 提示：")
    print("   - Cohere Rerank API 效果好，开箱即用")
    print("   - 支持多语言（包括中文）")
    print("   - 需要注册 Cohere 账号并获取 API Key")
    print("=" * 60)


if __name__ == "__main__":
    demo_cohere_rerank()
```

---

## ⚠️ 常见坑

1. **召回数量太少** — Rerank 需要足够的候选文档，建议召回 20-50 个
2. **直接对全部文档 Rerank** — 文档量大时计算成本极高，应该先召回再 Rerank
3. **忽略 Rerank 模型的语言支持** — 选择模型时要注意是否支持中文
4. **过度依赖 Rerank** — Rerank 不是万能的，如果召回阶段质量差，Rerank 也无法挽救
5. **不设置阈值过滤** — Rerank 后仍有低相关性文档，需要设置分数阈值过滤
6. **使用过时的模型** — Cross-Encoder 模型更新快，建议使用最新版本

---

## 🚀 实战建议

### 1. Rerank 使用场景判断

```
是否需要 Rerank？

检索效果已经很好（Top-3 准确率 > 90%）
  → 不需要 Rerank

检索效果一般（Top-3 准确率 70-90%）
  → 使用 Cross-Encoder Rerank

对检索质量要求极高
  → 召回更多 + Cross-Encoder Rerank + 阈值过滤

预算有限，追求速度
  → 不使用 Rerank，优化召回阶段

预算充足，追求效果
  → 使用 Cohere Rerank API 或 LLM Rerank
```

### 2. 参数调优建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `retrieve_k` | 20-50 | 召回文档数量 |
| `rerank_k` | 3-5 | 最终返回数量 |
| `rerank_threshold` | 0.6-0.8 | Rerank 分数阈值 |

### 3. 模型选择建议

| 场景 | 推荐模型 |
|------|---------|
| 英文通用 | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| 英文高性能 | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| 中文通用 | `cross-encoder/ms-marco-MiniLM-L-6-v2`（支持中文） |
| 多语言 | `Cohere rerank-multilingual-v2.0` |

---

## 📝 小练习

**代码练习：**

1. 运行 `rerank_basic.py`，理解 Rerank 的基本概念

2. 运行 `rerank_cross_encoder.py`，观察 Cross-Encoder 的效果

3. **思考题**：
   - 为什么不能直接用 Cross-Encoder 检索全部文档？
   - Rerank 能否完全替代召回阶段？
   - 如何评估 Rerank 的效果？

4. **进阶练习**：实现一个完整的 RAG 检索流程：向量召回 → Cross-Encoder Rerank → 阈值过滤

---

## 📚 本章知识点汇总

| 知识点名称 | 知识点类型 | 知识点介绍 | 参考链接 |
|-----------|-----------|-----------|---------|
| Rerank（重排序） | 概念 | 对检索结果进行二次排序，提升相关性 | - |
| 召回（Recall） | 概念 | 第一阶段检索，目标是尽可能多地召回相关文档 | - |
| 精排（Rerank） | 概念 | 第二阶段排序，目标是对召回文档精确排序 | - |
| Cross-Encoder | 技术 | 将 Query 和 Doc 一起输入模型，输出相关性分数，精度高但速度慢 | [Sentence-Transformers](https://www.sbert.net/examples/applications/cross-encoder/README.html) |
| Bi-Encoder | 技术 | Query 和 Doc 分别编码，速度快但精度较低，用于召回阶段 | [Sentence-Transformers](https://www.sbert.net/) |
| Two-Stage Retrieval | 架构 | 两阶段检索架构：召回（Bi-Encoder）→ 精排（Cross-Encoder） | - |
| Cohere Rerank API | 工具 | Cohere 提供的云端 Rerank 服务，开箱即用，支持多语言 | [Cohere Rerank](https://docs.cohere.com/docs/reranking) |
| LLM Rerank | 技术 | 用大模型（GPT-4）判断文档相关性，效果最好但成本最高 | - |
| ms-marco | 数据集 | Microsoft MARCO 数据集，常用于训练 Rerank 模型 | [MS MARCO](https://microsoft.github.io/msmarco/) |

---

## 📁 相关文件

| 文件名 | 说明 |
|--------|------|
| `README.md` | 本章学习笔记 |
| `rerank_basic.py` | 基础 Rerank 示例 |
| `rerank_cross_encoder.py` | Cross-Encoder Rerank 示例 |
| `rerank_pipeline.py` | 完整流程：检索 + Rerank |
| `rerank_cohere.py` | Cohere Rerank API 示例（可选） |
