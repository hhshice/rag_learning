# ⑤ 检索流程（Query → Recall）

## 🧠 概念解释

### 什么是检索（Retrieval）？

> **检索 = 根据用户问题，从向量数据库中找到最相关的文档片段**

这是 RAG 流程中最核心的环节，直接影响最终回答的质量。

```
用户问题 → [Embedding] → 查询向量 → [向量数据库] → 相关文档片段
```

### 检索 vs 召回（Recall）

| 概念 | 定义 | 目标 |
|------|------|------|
| **检索（Retrieval）** | 整个查找过程 | 找到相关文档 |
| **召回（Recall）** | 确保相关文档不遗漏 | 宁可多召回，不能漏掉 |

> **召回率 = 召回的相关文档数 / 所有相关文档总数**

### 检索的核心步骤

```
步骤 1: Query 预处理
    └─ 清洗、改写、扩展

步骤 2: Query Embedding
    └─ 将问题转换为向量

步骤 3: 向量检索
    └─ 在向量数据库中搜索相似文档

步骤 4: 结果过滤与排序
    └─ 根据相似度、元数据过滤结果

步骤 5: 返回 Top-K 结果
    └─ 返回最相关的 K 个文档片段
```

### 为什么需要 Query 预处理？

用户的问题往往不完美：

| 问题类型 | 示例 | 处理方法 |
|---------|------|---------|
| **模糊问题** | "这个怎么用？" | 上下文补全、改写 |
| **多意图问题** | "Python 的优缺点是什么？怎么安装？" | 拆分为多个子问题 |
| **关键词缺失** | "那个框架的教程" | 补充关键词 |
| **错别字** | "Pythn 教程" | 纠错 |

### 检索策略对比

| 策略 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| **纯向量检索** | 用 Query 向量找相似文档 | 语义理解好 | 可能遗漏关键词 | 通用场景 |
| **关键词检索** | BM25 等传统方法 | 精确匹配好 | 无法理解语义 | 专业术语检索 |
| **混合检索** | 向量 + 关键词结合 | 综合效果好 | 复杂度高 | 生产环境推荐 |
| **多路召回** | 多个检索器并行召回 | 召回率高 | 需要去重合并 | 高精度需求 |

### 检索参数调优

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `top_k` | 返回结果数量 | 3-10 |
| `score_threshold` | 相似度阈值 | 0.7-0.8 |
| `filter` | 元数据过滤 | 按业务需求 |
| `nprobe`（FAISS） | 搜索聚类数量 | 10-20 |

---

## 📦 示例代码

### 示例 1：基础检索流程

```python
# retrieval_basic.py
# 基础检索流程示例

import chromadb


def basic_retrieval():
    """最基础的检索流程"""
    
    # 1. 初始化向量数据库
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="knowledge_base")
    
    # 2. 准备知识库数据
    documents = [
        "Python 是一门流行的编程语言，语法简洁，易于学习。",
        "RAG 是检索增强生成技术，让大模型能够基于外部知识回答问题。",
        "向量数据库用于存储和检索文本的向量表示。",
        "Embedding 是将文本转换为向量的技术。",
        "FastAPI 是一个现代的 Python Web 框架，高性能且易用。",
        "机器学习是人工智能的一个分支，使用算法从数据中学习。",
        "深度学习使用多层神经网络进行特征学习。",
        "自然语言处理让计算机能够理解和生成人类语言。"
    ]
    
    # 添加文档（如果集合为空）
    if collection.count() == 0:
        collection.upsert(
            ids=[f"doc_{i}" for i in range(len(documents))],
            documents=documents
        )
    
    # 3. 执行检索
    query = "什么是 RAG？"
    
    results = collection.query(
        query_texts=[query],
        n_results=3  # 返回 top-3
    )
    
    # 4. 显示结果
    print("=" * 60)
    print("🔍 基础检索示例")
    print("=" * 60)
    print(f"查询：{query}")
    print(f"返回数量：{len(results['documents'][0])}")
    print("=" * 60)
    
    for i, (doc, distance) in enumerate(
        zip(results['documents'][0], results['distances'][0]), 1
    ):
        similarity = 1 - distance
        print(f"\n[{i}] 相似度: {similarity:.4f}")
        print(f"    {doc}")
    
    return results


if __name__ == "__main__":
    basic_retrieval()
```

---

### 示例 2：Query 预处理与改写

```python
# query_preprocessing.py
# Query 预处理示例：清洗、改写、扩展

import re


def clean_query(query):
    """
    清洗查询文本
    
    - 去除多余空格
    - 去除特殊字符
    - 转换为小写（可选）
    """
    # 去除首尾空格
    query = query.strip()
    
    # 去除多余空格
    query = re.sub(r'\s+', ' ', query)
    
    # 去除特殊字符（保留中文、英文、数字、标点）
    query = re.sub(r'[^\w\s\u4e00-\u9fff，。！？、；：]', '', query)
    
    return query


def expand_query(query, synonyms):
    """
    查询扩展：添加同义词
    
    Args:
        query: 原始查询
        synonyms: 同义词字典，如 {"Python": ["Python", "python", "py"]}
    
    Returns:
        expanded_query: 扩展后的查询
    """
    words = query.split()
    expanded_words = []
    
    for word in words:
        expanded_words.append(word)
        # 添加同义词
        if word in synonyms:
            expanded_words.extend(synonyms[word])
    
    return ' '.join(expanded_words)


def rewrite_query(query, context=None):
    """
    查询改写：补充上下文信息
    
    Args:
        query: 原始查询
        context: 上下文信息（如用户历史对话、当前场景等）
    
    Returns:
        rewritten_query: 改写后的查询
    """
    # 示例：如果查询模糊，补充上下文
    vague_keywords = ["这个", "那个", "它", "怎么用", "怎么处理"]
    
    if any(keyword in query for keyword in vague_keywords):
        if context:
            query = f"{context}：{query}"
        else:
            query = f"请详细解释：{query}"
    
    return query


def split_multi_intent_query(query):
    """
    拆分多意图查询
    
    Args:
        query: 包含多个意图的查询
    
    Returns:
        sub_queries: 拆分后的子查询列表
    """
    # 使用标点符号拆分
    separators = ["？", "？", "，", "；", "和", "以及", "还有"]
    
    sub_queries = [query]
    for sep in separators:
        new_queries = []
        for q in sub_queries:
            parts = q.split(sep)
            new_queries.extend([p.strip() for p in parts if p.strip()])
        sub_queries = new_queries
    
    return sub_queries


def demo_query_preprocessing():
    """演示 Query 预处理"""
    
    print("=" * 60)
    print("🔧 Query 预处理演示")
    print("=" * 60)
    
    # 1. 清洗
    raw_query = "  Python   是什么？ ！！！  "
    cleaned = clean_query(raw_query)
    print(f"\n【清洗】")
    print(f"原始: '{raw_query}'")
    print(f"清洗后: '{cleaned}'")
    
    # 2. 查询扩展
    query = "Python 教程"
    synonyms = {
        "Python": ["python", "py"],
        "教程": ["教程", "入门", "学习"]
    }
    expanded = expand_query(query, synonyms)
    print(f"\n【查询扩展】")
    print(f"原始: '{query}'")
    print(f"扩展后: '{expanded}'")
    
    # 3. 查询改写
    vague_query = "这个怎么用？"
    context = "FastAPI 框架"
    rewritten = rewrite_query(vague_query, context)
    print(f"\n【查询改写】")
    print(f"原始: '{vague_query}'")
    print(f"改写后: '{rewritten}'")
    
    # 4. 多意图拆分
    multi_query = "Python 的优缺点是什么？怎么安装？"
    sub_queries = split_multi_intent_query(multi_query)
    print(f"\n【多意图拆分】")
    print(f"原始: '{multi_query}'")
    print(f"拆分后: {sub_queries}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_query_preprocessing()
```

---

### 示例 3：混合检索（向量 + 关键词）

```python
# hybrid_retrieval.py
# 混合检索示例：向量检索 + 关键词检索

import chromadb
from collections import Counter
import re


def simple_keyword_search(documents, query, top_k=3):
    """
    简单的关键词检索（BM25 简化版）
    
    Args:
        documents: 文档列表
        query: 查询文本
        top_k: 返回结果数量
    
    Returns:
        results: 排序后的文档索引和分数
    """
    # 分词（简单实现）
    def tokenize(text):
        return re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text.lower())
    
    query_tokens = tokenize(query)
    query_token_set = set(query_tokens)
    
    scores = []
    for i, doc in enumerate(documents):
        doc_tokens = tokenize(doc)
        doc_token_set = set(doc_tokens)
        
        # 计算重叠词数量
        overlap = len(query_token_set & doc_token_set)
        
        # 计算词频
        doc_counter = Counter(doc_tokens)
        tf = sum(doc_counter[token] for token in query_tokens)
        
        # 简化分数：重叠数 + 词频
        score = overlap + tf * 0.1
        scores.append((i, score))
    
    # 排序
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores[:top_k]


def hybrid_retrieval(collection, documents, query, top_k=3, alpha=0.5):
    """
    混合检索：向量检索 + 关键词检索
    
    Args:
        collection: Chroma 集合对象
        documents: 文档列表
        query: 查询文本
        top_k: 返回结果数量
        alpha: 向量检索权重（0-1），关键词检索权重为 1-alpha
    
    Returns:
        results: 混合排序后的结果
    """
    # 1. 向量检索
    vector_results = collection.query(
        query_texts=[query],
        n_results=top_k * 2  # 多召回一些
    )
    
    vector_scores = {}
    for doc, distance in zip(vector_results['ids'][0], vector_results['distances'][0]):
        vector_scores[doc] = 1 - distance  # 转换为相似度
    
    # 2. 关键词检索
    keyword_results = simple_keyword_search(documents, query, top_k=top_k * 2)
    
    keyword_scores = {}
    for idx, score in keyword_results:
        doc_id = f"doc_{idx}"
        keyword_scores[doc_id] = score
    
    # 3. 归一化分数
    def normalize(scores):
        if not scores:
            return {}
        max_score = max(scores.values()) if scores else 1
        min_score = min(scores.values()) if scores else 0
        range_score = max_score - min_score if max_score != min_score else 1
        return {k: (v - min_score) / range_score for k, v in scores.items()}
    
    vector_scores_norm = normalize(vector_scores)
    keyword_scores_norm = normalize(keyword_scores)
    
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


def demo_hybrid_retrieval():
    """演示混合检索"""
    
    print("=" * 60)
    print("🔀 混合检索演示")
    print("=" * 60)
    
    # 准备数据
    documents = [
        "Python 是一门流行的编程语言，语法简洁。",
        "Python 的列表（List）是一种有序集合。",
        "RAG 是检索增强生成技术。",
        "Python 安装教程：访问官网下载安装包。",
        "FastAPI 是 Python Web 框架。",
        "机器学习使用算法从数据中学习模式。"
    ]
    
    # 初始化向量数据库
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="hybrid_demo")
    
    # 添加文档
    if collection.count() == 0:
        collection.upsert(
            ids=[f"doc_{i}" for i in range(len(documents))],
            documents=documents
        )
    
    # 测试查询
    query = "Python 怎么安装"
    
    print(f"\n查询：{query}")
    print("=" * 60)
    
    # 1. 纯向量检索
    print("\n【纯向量检索】")
    vector_results = collection.query(
        query_texts=[query],
        n_results=3
    )
    for i, (doc, dist) in enumerate(zip(vector_results['documents'][0], vector_results['distances'][0]), 1):
        print(f"  [{i}] 相似度: {1-dist:.4f} - {doc[:40]}")
    
    # 2. 混合检索
    print("\n【混合检索】")
    hybrid_results = hybrid_retrieval(collection, documents, query, top_k=3, alpha=0.5)
    
    for i, (doc_id, score) in enumerate(hybrid_results, 1):
        idx = int(doc_id.split('_')[1])
        print(f"  [{i}] 分数: {score:.4f} - {documents[idx][:40]}")
    
    print("\n" + "=" * 60)
    print("💡 混合检索结合了语义理解和关键词匹配，效果更好")
    print("=" * 60)


if __name__ == "__main__":
    demo_hybrid_retrieval()
```

---

### 示例 4：带过滤的高级检索

```python
# advanced_retrieval.py
# 高级检索示例：元数据过滤、相似度阈值、重排序

import chromadb


def advanced_retrieval():
    """演示高级检索功能"""
    
    print("=" * 60)
    print("🎯 高级检索示例")
    print("=" * 60)
    
    # 初始化向量数据库
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="advanced_retrieval")
    
    # 准备数据（带元数据）
    documents = [
        {
            "id": "doc_1",
            "content": "Python 是一门流行的编程语言，适合初学者。",
            "metadata": {"category": "编程", "level": "入门", "author": "张三"}
        },
        {
            "id": "doc_2",
            "content": "Python 高级特性：装饰器、生成器、上下文管理器。",
            "metadata": {"category": "编程", "level": "高级", "author": "李四"}
        },
        {
            "id": "doc_3",
            "content": "RAG 入门：检索增强生成技术基础概念。",
            "metadata": {"category": "AI", "level": "入门", "author": "张三"}
        },
        {
            "id": "doc_4",
            "content": "RAG 进阶：向量数据库选型与优化策略。",
            "metadata": {"category": "AI", "level": "高级", "author": "王五"}
        },
        {
            "id": "doc_5",
            "content": "FastAPI 快速入门：构建第一个 RESTful API。",
            "metadata": {"category": "Web开发", "level": "入门", "author": "张三"}
        }
    ]
    
    # 添加文档
    if collection.count() == 0:
        collection.upsert(
            ids=[doc["id"] for doc in documents],
            documents=[doc["content"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents]
        )
    
    # 1. 带元数据过滤的检索
    print("\n【1. 元数据过滤】")
    query = "入门教程"
    
    # 只检索 level="入门" 的文档
    filtered_results = collection.query(
        query_texts=[query],
        n_results=5,
        where={"level": "入门"}  # 元数据过滤
    )
    
    print(f"查询：{query}（过滤条件：level='入门'）")
    for i, (doc, meta) in enumerate(
        zip(filtered_results['documents'][0], filtered_results['metadatas'][0]), 1
    ):
        print(f"  [{i}] [{meta['category']}] {doc[:40]}")
    
    # 2. 相似度阈值过滤
    print("\n【2. 相似度阈值过滤】")
    query = "深度学习神经网络"
    
    results = collection.query(
        query_texts=[query],
        n_results=5
    )
    
    threshold = 0.5  # 相似度阈值
    
    print(f"查询：{query}（阈值：{threshold}）")
    filtered_by_threshold = []
    
    for doc, dist, meta in zip(
        results['documents'][0],
        results['distances'][0],
        results['metadatas'][0]
    ):
        similarity = 1 - dist
        if similarity >= threshold:
            filtered_by_threshold.append((doc, similarity, meta))
            print(f"  ✅ [{similarity:.4f}] {doc[:40]}")
        else:
            print(f"  ❌ [{similarity:.4f}] {doc[:40]} (低于阈值)")
    
    # 3. 多条件过滤
    print("\n【3. 多条件过滤】")
    query = "教程"
    
    # 使用 $and 进行多条件过滤
    multi_filter_results = collection.query(
        query_texts=[query],
        n_results=5,
        where={
            "$and": [
                {"level": "入门"},
                {"author": "张三"}
            ]
        }
    )
    
    print(f"查询：{query}（过滤条件：level='入门' AND author='张三'）")
    for i, (doc, meta) in enumerate(
        zip(multi_filter_results['documents'][0], multi_filter_results['metadatas'][0]), 1
    ):
        print(f"  [{i}] {doc[:40]}")
    
    # 4. 组合检索策略
    print("\n【4. 组合检索策略】")
    
    def retrieve_with_strategy(collection, query, filters=None, threshold=0.6, top_k=5):
        """
        组合检索策略
        
        1. 先用元数据过滤缩小范围
        2. 再用向量检索找相似文档
        3. 最后用阈值过滤低质量结果
        """
        results = collection.query(
            query_texts=[query],
            n_results=top_k * 2,
            where=filters
        )
        
        # 阈值过滤
        filtered = []
        for i, (doc_id, doc, dist, meta) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0]
        )):
            similarity = 1 - dist
            if similarity >= threshold:
                filtered.append({
                    "id": doc_id,
                    "content": doc,
                    "similarity": similarity,
                    "metadata": meta
                })
        
        return filtered[:top_k]
    
    query = "编程入门"
    results = retrieve_with_strategy(
        collection,
        query,
        filters={"category": "编程"},
        threshold=0.5,
        top_k=3
    )
    
    print(f"查询：{query}")
    print(f"策略：元数据过滤(category='编程') + 阈值过滤(0.5)")
    print("\n最终结果：")
    for i, result in enumerate(results, 1):
        print(f"  [{i}] 相似度: {result['similarity']:.4f}")
        print(f"      {result['content']}")
        print(f"      元数据: {result['metadata']}")
    
    print("\n" + "=" * 60)
    print("✅ 高级检索演示完成")
    print("=" * 60)


if __name__ == "__main__":
    advanced_retrieval()
```

---

## ⚠️ 常见坑

1. **忽略 Query 预处理** — 用户问题可能模糊、有错别字、包含多意图，直接检索效果差
2. **只用向量检索** — 对于包含专业术语、精确关键词的查询，纯向量检索可能遗漏
3. **top_k 设置不当** — 太小会漏掉相关文档，太大会引入噪音
4. **不设置相似度阈值** — 低质量结果混入，影响最终回答
5. **忽略元数据过滤** — 业务场景中，元数据过滤能显著提升检索精度
6. **不考虑检索性能** — 大规模数据下，需要优化索引参数（如 FAISS 的 nprobe）

---

## 🚀 实战建议

### 1. 检索流程优化清单

```
□ Query 预处理
  ├─ 清洗（去空格、去特殊字符）
  ├─ 改写（补充上下文）
  └─ 扩展（添加同义词）

□ 检索策略选择
  ├─ 通用场景 → 混合检索
  ├─ 专业术语 → 关键词权重提高
  └─ 语义理解 → 向量权重提高

□ 结果过滤
  ├─ 元数据过滤（按业务需求）
  ├─ 相似度阈值（0.6-0.8）
  └─ 去重（避免重复内容）

□ 性能优化
  ├─ 调整索引参数（nprobe）
  ├─ 批量检索
  └─ 缓存热门查询
```

### 2. 参数调优建议

| 场景 | top_k | threshold | alpha（混合检索） |
|------|-------|-----------|------------------|
| 高精度需求 | 3-5 | 0.7-0.8 | 0.6 |
| 高召回需求 | 10-20 | 0.5-0.6 | 0.4 |
| 平衡场景 | 5-10 | 0.6-0.7 | 0.5 |

### 3. 检索效果评估指标

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| **召回率（Recall）** | 召回的相关文档占所有相关文档的比例 | 召回的相关文档数 / 总相关文档数 |
| **精确率（Precision）** | 召回的文档中相关文档的比例 | 召回的相关文档数 / 召回的总文档数 |
| **MRR** | 第一个相关文档的排名倒数的平均值 | 1 / rank |
| **NDCG** | 考虑排序位置的归一化指标 | 复杂公式 |

---

## 📝 小练习

**代码练习：**

1. 运行 `retrieval_basic.py`，观察检索结果，尝试修改查询语句

2. 实现 Query 扩展功能：给定一个查询，使用同义词字典扩展查询词

3. **思考题**：
   - 如何评估检索效果的好坏？
   - 向量检索和关键词检索各有什么优缺点？
   - 在什么场景下应该使用混合检索？

4. **进阶练习**：实现一个检索器，支持用户自定义过滤条件（如按时间、作者、类别过滤）

---

## 📚 本章知识点汇总

| 知识点名称 | 知识点类型 | 知识点介绍 | 参考链接 |
|-----------|-----------|-----------|---------|
| 检索（Retrieval） | 概念 | 根据查询从向量数据库中找到相关文档的过程 | - |
| 召回（Recall） | 概念 | 确保相关文档不被遗漏，召回率 = 召回的相关文档数 / 总相关文档数 | - |
| Query 预处理 | 技术 | 对用户查询进行清洗、改写、扩展，提升检索效果 | - |
| 查询扩展 | 技术 | 为查询添加同义词、相关词，扩大检索范围 | - |
| 混合检索 | 技术 | 结合向量检索和关键词检索的综合检索方法 | - |
| BM25 | 算法 | 经典的关键词检索算法，基于词频和文档频率 | [BM25 维基百科](https://en.wikipedia.org/wiki/Okapi_BM25) |
| 元数据过滤 | 技术 | 根据文档的元数据（如类别、时间、作者）过滤检索结果 | [Chroma 过滤文档](https://docs.trychroma.com/usage-guide#using-where-filters) |
| 相似度阈值 | 技术参数 | 过滤低质量检索结果的相似度下限，通常为 0.6-0.8 | - |
| top_k | 技术参数 | 检索返回的文档数量，通常为 3-10 | - |
| MRR（Mean Reciprocal Rank） | 评估指标 | 第一个相关文档排名的倒数平均值，评估检索排序质量 | - |

---

## 📁 相关文件

| 文件名 | 说明 |
|--------|------|
| `README.md` | 本章学习笔记 |
| `retrieval_basic.py` | 基础检索流程示例 |
| `query_preprocessing.py` | Query 预处理示例（清洗、改写、扩展） |
| `hybrid_retrieval.py` | 混合检索示例（向量 + 关键词） |
| `advanced_retrieval.py` | 高级检索示例（元数据过滤、阈值过滤） |
