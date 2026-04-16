# 第9阶段：优化方向（召回率、精度、速度）

## 🎯 学习目标

- 掌握 RAG 系统的核心评估指标
- 了解召回率、精度、速度的权衡关系
- 学会针对性的优化策略
- 掌握性能测试和调优方法

---

## 📊 核心指标对比

| 指标 | 定义 | 计算方式 | 目标值 |
|------|------|---------|--------|
| **召回率 (Recall)** | 召回多少相关文档 | 相关文档被召回 / 总相关文档 | > 80% |
| **精度 (Precision)** | 召回文档中有多少相关 | 相关召回文档 / 总召回文档 | > 70% |
| **MRR** | 第一个相关文档的位置倒数 | 1 / 第一个相关文档排名 | > 0.7 |
| **延迟 (Latency)** | 从查询到返回结果的时间 | P50/P95/P99 延迟 | < 500ms |
| **吞吐量 (QPS)** | 每秒处理请求数 | 请求数 / 时间 | 根据业务定 |

---

## 📦 知识点表格

| 知识点名称 | 知识点类型 | 知识点介绍 | 参考链接 |
|-----------|-----------|-----------|---------|
| 召回率优化 | 技术 | 通过调整检索策略提高相关文档召回率 | [向量检索优化](https://github.com/facebookresearch/faiss/wiki) |
| 精度优化 | 技术 | 通过 Rerank、元数据过滤提高检索精度 | [Rerank 策略](./06_Rerank二次排序/) |
| 速度优化 | 技术 | 通过索引优化、并行处理降低延迟 | [FAISS 性能调优](https://github.com/facebookresearch/faiss/wiki/Guidelines-for-choosing-an-index) |
| 评估指标 | 概念 | 用于衡量 RAG 系统效果的标准 | [RAGAS](https://github.com/explodinggradients/ragas) |
| 混合检索 | 技术 | 结合向量检索和关键词检索 | [Hybrid Search](https://www.pinecone.io/learn/hybrid-search/) |
| 索引优化 | 技术 | HNSW、IVF 等索引结构优化 | [FAISS Index](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) |
| 批处理 | 技术 | 批量 Embedding 和检索提升吞吐量 | [Batch Processing](https://huggingface.co/docs/transformers/main_classes/pipelines) |
| 缓存策略 | 技术 | 缓存热门 Query 的检索结果 | [Caching Patterns](https://redis.io/docs/manual/patterns/) |
| 查询扩展 | 技术 | 通过改写、扩展 Query 提高召回 | [Query Expansion](https://nlp.stanford.edu/IR-book/html/htmledition/query-expansion-1.html) |
| 负采样 | 技术 | 在训练时加入负例提升判别能力 | [Negative Sampling](https://arxiv.org/abs/1402.3722) |

---

## 🧠 核心概念解释

### 1. 召回率 vs 精度的权衡

```
召回率 ↑ → 召回更多文档 → 精度可能 ↓（混入无关文档）
精度 ↑   → 召回更准确   → 召回率可能 ↓（漏掉相关文档）

理想状态：高召回率 + 高精度
现实：需要权衡，根据业务场景调整
```

**业务场景示例：**
- 法律咨询：优先精度（不能给错误答案）
- 内容推荐：优先召回率（宁可多推，不要漏掉）
- 医疗问答：精度 + 召回率都要高（错误代价大）

---

### 2. 速度优化三要素

```
┌─────────────────────────────────────────────┐
│           RAG 速度瓶颈                       │
├─────────────────────────────────────────────┤
│ ① Embedding 耗时   → 批处理 + GPU 加速      │
│ ② 向量检索耗时     → 索引优化 (HNSW/IVF)    │
│ ③ Rerank 耗时      → 并行处理 + 减少 Doc 数 │
│ ④ LLM 生成耗时     → 流式输出 + 缓存        │
└─────────────────────────────────────────────┘
```

---

## 📦 示例代码

### 示例 1：召回率优化 - 混合检索

```python
# optimize_retrieval.py
# 召回率优化策略：混合检索 + Query 扩展

import chromadb
from typing import List, Dict, Set

def hybrid_search(
    query: str,
    vector_results: List[Dict],
    keyword_results: List[Dict],
    alpha: float = 0.7  # 向量检索权重
) -> List[Dict]:
    """
    混合检索：融合向量检索和关键词检索结果
    alpha: 向量检索权重，(1-alpha) 为关键词检索权重
    """
    # 合并结果
    all_docs = {}
    
    # 向量检索结果
    for i, doc in enumerate(vector_results):
        doc_id = doc['id']
        score = 1 / (i + 1)  # 排名倒数
        all_docs[doc_id] = {
            'doc': doc,
            'vector_score': score,
            'keyword_score': 0
        }
    
    # 关键词检索结果
    for i, doc in enumerate(keyword_results):
        doc_id = doc['id']
        score = 1 / (i + 1)
        if doc_id in all_docs:
            all_docs[doc_id]['keyword_score'] = score
        else:
            all_docs[doc_id] = {
                'doc': doc,
                'vector_score': 0,
                'keyword_score': score
            }
    
    # 计算混合分数
    for doc_id in all_docs:
        all_docs[doc_id]['final_score'] = (
            alpha * all_docs[doc_id]['vector_score'] +
            (1 - alpha) * all_docs[doc_id]['keyword_score']
        )
    
    # 按混合分数排序
    sorted_results = sorted(
        all_docs.values(),
        key=lambda x: x['final_score'],
        reverse=True
    )
    
    return [item['doc'] for item in sorted_results]


def query_expansion(query: str, synonyms: Dict[str, List[str]]) -> List[str]:
    """
    Query 扩展：将原始 Query 扩展为多个相关 Query
    """
    queries = [query]
    
    # 同义词扩展
    words = query.split()
    for word in words:
        if word in synonyms:
            for synonym in synonyms[word]:
                expanded_query = query.replace(word, synonym)
                queries.append(expanded_query)
    
    return queries


# 使用示例
if __name__ == "__main__":
    # 模拟检索结果
    vector_results = [
        {'id': 'doc1', 'text': 'RAG 是检索增强生成'},
        {'id': 'doc2', 'text': '向量数据库是 RAG 的核心'},
        {'id': 'doc3', 'text': 'Embedding 将文本转为向量'}
    ]
    
    keyword_results = [
        {'id': 'doc1', 'text': 'RAG 是检索增强生成'},
        {'id': 'doc4', 'text': 'RAG 系统架构设计'},
        {'id': 'doc5', 'text': 'RAG 应用案例'}
    ]
    
    # 混合检索
    results = hybrid_search(
        query="RAG 是什么",
        vector_results=vector_results,
        keyword_results=keyword_results,
        alpha=0.7
    )
    
    print("混合检索结果：")
    for i, doc in enumerate(results[:5], 1):
        print(f"{i}. {doc['id']}: {doc['text']}")
```

---

### 示例 2：精度优化 - Rerank + 过滤

```python
# optimize_precision.py
# 精度优化策略：Rerank + 元数据过滤 + 阈值控制

from sentence_transformers import CrossEncoder
from typing import List, Dict
import numpy as np


class PrecisionOptimizer:
    """精度优化器"""
    
    def __init__(self, rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.reranker = CrossEncoder(rerank_model)
    
    def metadata_filter(
        self,
        results: List[Dict],
        filters: Dict
    ) -> List[Dict]:
        """
        元数据过滤：根据元数据筛选文档
        filters: {'category': '技术', 'year': 2024}
        """
        filtered = []
        for doc in results:
            match = True
            for key, value in filters.items():
                if doc.get('metadata', {}).get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(doc)
        return filtered
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Cross-Encoder Rerank：精确重排序
        """
        # 构造输入对
        pairs = [(query, doc['text']) for doc in documents]
        
        # 计算相关性分数
        scores = self.reranker.predict(pairs)
        
        # 按分数排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 返回 top_k
        return [
            {**doc, 'rerank_score': float(score)}
            for doc, score in scored_docs[:top_k]
        ]
    
    def threshold_filter(
        self,
        results: List[Dict],
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        阈值过滤：只保留分数高于阈值的文档
        """
        return [
            doc for doc in results
            if doc.get('rerank_score', 0) >= threshold
        ]
    
    def optimize(
        self,
        query: str,
        candidates: List[Dict],
        filters: Dict = None,
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        完整的精度优化流程
        """
        # 1. 元数据过滤
        if filters:
            candidates = self.metadata_filter(candidates, filters)
        
        # 2. Rerank
        reranked = self.rerank(query, candidates, top_k=top_k)
        
        # 3. 阈值过滤
        filtered = self.threshold_filter(reranked, threshold)
        
        return filtered


# 使用示例
if __name__ == "__main__":
    optimizer = PrecisionOptimizer()
    
    # 模拟检索结果
    candidates = [
        {
            'id': 'doc1',
            'text': 'RAG 是检索增强生成技术',
            'metadata': {'category': '技术', 'year': 2024}
        },
        {
            'id': 'doc2',
            'text': '向量数据库用于存储向量',
            'metadata': {'category': '技术', 'year': 2023}
        },
        {
            'id': 'doc3',
            'text': 'Python 是一门编程语言',
            'metadata': {'category': '编程', 'year': 2024}
        }
    ]
    
    # 精度优化
    results = optimizer.optimize(
        query="什么是 RAG",
        candidates=candidates,
        filters={'category': '技术'},
        top_k=3,
        threshold=0.3
    )
    
    print("精度优化结果：")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc['id']}: {doc['text']}")
        print(f"   分数: {doc['rerank_score']:.3f}")
```

---

### 示例 3：速度优化 - 索引 + 并行

```python
# optimize_speed.py
# 速度优化策略：索引优化 + 批处理 + 并行

import time
import faiss
import numpy as np
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
import chromadb


class SpeedOptimizer:
    """速度优化器"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = None
    
    # ========== 索引优化 ==========
    
    def build_index_flat(self, embeddings: np.ndarray):
        """
        Flat 索引：暴力搜索，精度最高，速度最慢
        适合：小规模数据（< 10万）
        """
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        return self.index
    
    def build_index_ivf(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 100
    ):
        """
        IVF 索引：倒排索引，速度快，精度略有下降
        适合：中大规模数据（10万 - 1000万）
        """
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        self.index = faiss.IndexIVFFlat(
            quantizer,
            self.embedding_dim,
            n_clusters
        )
        
        # 训练
        self.index.train(embeddings.astype('float32'))
        self.index.add(embeddings.astype('float32'))
        return self.index
    
    def build_index_hnsw(
        self,
        embeddings: np.ndarray,
        m: int = 32,
        ef_search: int = 64
    ):
        """
        HNSW 索引：图索引，速度极快，精度高
        适合：大规模数据（> 100万），内存充足
        """
        self.index = faiss.IndexHNSWFlat(
            self.embedding_dim,
            m,
            faiss.METRIC_INNER_PRODUCT
        )
        self.index.hnsw.efSearch = ef_search
        self.index.add(embeddings.astype('float32'))
        return self.index
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """向量检索"""
        query = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query, k)
        return distances[0], indices[0]
    
    # ========== 批处理优化 ==========
    
    @staticmethod
    def batch_embedding(
        texts: List[str],
        model,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        批量 Embedding：提升吞吐量
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = model.encode(batch)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    # ========== 并行处理 ==========
    
    @staticmethod
    def parallel_rerank(
        query: str,
        documents: List[dict],
        rerank_func,
        n_workers: int = 4
    ) -> List[dict]:
        """
        并行 Rerank：多线程处理
        """
        def process_chunk(chunk):
            return rerank_func(query, chunk)
        
        # 分块
        chunk_size = len(documents) // n_workers
        chunks = [
            documents[i:i + chunk_size]
            for i in range(0, len(documents), chunk_size)
        ]
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(process_chunk, chunks))
        
        # 合并结果
        all_results = []
        for chunk_result in results:
            all_results.extend(chunk_result)
        
        return all_results


# 使用示例
if __name__ == "__main__":
    optimizer = SpeedOptimizer(embedding_dim=128)
    
    # 生成模拟数据
    n_docs = 10000
    embeddings = np.random.randn(n_docs, 128).astype('float32')
    
    print("=" * 60)
    print("索引性能对比测试")
    print("=" * 60)
    
    # 测试 Flat 索引
    start = time.time()
    optimizer.build_index_flat(embeddings)
    build_time_flat = time.time() - start
    
    start = time.time()
    query = np.random.randn(128)
    distances, indices = optimizer.search(query, k=5)
    search_time_flat = time.time() - start
    
    print(f"\nFlat 索引:")
    print(f"  构建时间: {build_time_flat:.3f}s")
    print(f"  检索时间: {search_time_flat*1000:.2f}ms")
    
    # 测试 IVF 索引
    start = time.time()
    optimizer.build_index_ivf(embeddings, n_clusters=100)
    build_time_ivf = time.time() - start
    
    start = time.time()
    distances, indices = optimizer.search(query, k=5)
    search_time_ivf = time.time() - start
    
    print(f"\nIVF 索引:")
    print(f"  构建时间: {build_time_ivf:.3f}s")
    print(f"  检索时间: {search_time_ivf*1000:.2f}ms")
    
    # 测试 HNSW 索引
    start = time.time()
    optimizer.build_index_hnsw(embeddings, m=32)
    build_time_hnsw = time.time() - start
    
    start = time.time()
    distances, indices = optimizer.search(query, k=5)
    search_time_hnsw = time.time() - start
    
    print(f"\nHNSW 索引:")
    print(f"  构建时间: {build_time_hnsw:.3f}s")
    print(f"  检索时间: {search_time_hnsw*1000:.2f}ms")
```

---

### 示例 4：评估指标计算

```python
# optimize_metrics.py
# RAG 系统评估指标计算

from typing import List, Dict
import numpy as np


class RAGEvaluator:
    """RAG 评估器"""
    
    @staticmethod
    def calculate_recall(
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        计算召回率
        召回率 = 召回的相关文档 / 总相关文档
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)
        
        # 召回的相关文档
        retrieved_relevant = len(retrieved_set & relevant_set)
        
        return retrieved_relevant / len(relevant_set)
    
    @staticmethod
    def calculate_precision(
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        计算精度
        精度 = 召回的相关文档 / 总召回文档
        """
        if not retrieved_ids:
            return 0.0
        
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)
        
        retrieved_relevant = len(retrieved_set & relevant_set)
        
        return retrieved_relevant / len(retrieved_set)
    
    @staticmethod
    def calculate_mrr(
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        计算 MRR (Mean Reciprocal Rank)
        MRR = 1 / 第一个相关文档的排名
        """
        relevant_set = set(relevant_ids)
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def calculate_ndcg(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 10
    ) -> float:
        """
        计算 NDCG (Normalized Discounted Cumulative Gain)
        考虑排序位置的理想指标
        """
        relevant_set = set(relevant_ids)
        
        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # i+2 因为 log2(1) = 0
        
        # IDCG (理想情况)
        idcg = 0.0
        for i in range(min(len(relevant_ids), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate(
        self,
        query_results: Dict[str, List[str]],  # {query: [retrieved_ids]}
        ground_truth: Dict[str, List[str]],   # {query: [relevant_ids]}
        k_values: List[int] = [5, 10, 20]
    ) -> Dict:
        """
        完整评估
        """
        metrics = {
            'recall': {},
            'precision': {},
            'mrr': [],
            'ndcg': {}
        }
        
        # 初始化
        for k in k_values:
            metrics['recall'][f'recall@{k}'] = []
            metrics['precision'][f'precision@{k}'] = []
            metrics['ndcg'][f'ndcg@{k}'] = []
        
        # 逐 Query 计算
        for query, retrieved_ids in query_results.items():
            if query not in ground_truth:
                continue
            
            relevant_ids = ground_truth[query]
            
            # 各指标
            for k in k_values:
                metrics['recall'][f'recall@{k}'].append(
                    self.calculate_recall(retrieved_ids[:k], relevant_ids)
                )
                metrics['precision'][f'precision@{k}'].append(
                    self.calculate_precision(retrieved_ids[:k], relevant_ids)
                )
                metrics['ndcg'][f'ndcg@{k}'].append(
                    self.calculate_ndcg(retrieved_ids, relevant_ids, k)
                )
            
            metrics['mrr'].append(
                self.calculate_mrr(retrieved_ids, relevant_ids)
            )
        
        # 计算平均值
        result = {}
        for k in k_values:
            result[f'recall@{k}'] = np.mean(metrics['recall'][f'recall@{k}'])
            result[f'precision@{k}'] = np.mean(metrics['precision'][f'precision@{k}'])
            result[f'ndcg@{k}'] = np.mean(metrics['ndcg'][f'ndcg@{k}'])
        
        result['mrr'] = np.mean(metrics['mrr'])
        
        return result


# 使用示例
if __name__ == "__main__":
    evaluator = RAGEvaluator()
    
    # 模拟数据
    query_results = {
        'query1': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5'],
        'query2': ['doc3', 'doc1', 'doc6', 'doc7', 'doc8'],
        'query3': ['doc2', 'doc4', 'doc9', 'doc10', 'doc1']
    }
    
    ground_truth = {
        'query1': ['doc1', 'doc3', 'doc5'],
        'query2': ['doc1', 'doc3', 'doc6'],
        'query3': ['doc2', 'doc4', 'doc10']
    }
    
    # 评估
    results = evaluator.evaluate(query_results, ground_truth)
    
    print("=" * 60)
    print("RAG 系统评估结果")
    print("=" * 60)
    
    for metric, value in results.items():
        print(f"{metric:15s}: {value:.3f}")
```

---

## ⚠️ 常见坑

### 1. 只追求单一指标

```python
# ❌ 错误：只看召回率
召回率 95%，但精度只有 30%  → 返回大量无关文档，用户体验差

# ✅ 正确：综合考虑
召回率 85% + 精度 75%  → 平衡效果更好
```

### 2. 忽略延迟

```python
# ❌ 错误：追求极致精度
召回 100 个文档 → Rerank → 延迟 5 秒

# ✅ 正确：权衡延迟
召回 20 个文档 → Rerank → 延迟 500ms，精度损失 < 5%
```

### 3. 索引选择不当

```python
# ❌ 错误：小数据用 HNSW
数据量 1 万条 → HNSW 构建慢，内存浪费

# ✅ 正确：根据数据量选择
< 10 万条 → Flat 索引
10 万 - 1000 万 → IVF 索引
> 1000 万 → HNSW 索引
```

### 4. 评估数据不准确

```python
# ❌ 错误：人工标注不充分
只用 10 个 Query 评估 → 结果不可靠

# ✅ 正确：充分标注
至少 100-200 个 Query + 多人标注 + 一致性检验
```

### 5. 忽略缓存

```python
# ❌ 错误：每次查询都重新计算
相同 Query 多次查询 → 重复计算浪费资源

# ✅ 正确：添加缓存层
热门 Query 缓存结果 → 命中率 30-50%，延迟降低 80%
```

---

## 🚀 实战建议

### 优化优先级

```
第 1 步：建立评估体系
  ↓
第 2 步：召回率优化（混合检索、Query 扩展）
  ↓
第 3 步：精度优化（Rerank、过滤、阈值）
  ↓
第 4 步：速度优化（索引、批处理、缓存）
  ↓
第 5 步：持续监控和迭代
```

### 各阶段目标值

| 指标 | 基线 | 及格 | 优秀 |
|------|------|------|------|
| Recall@10 | 60% | 75% | 85% |
| Precision@5 | 50% | 70% | 80% |
| MRR | 0.5 | 0.7 | 0.85 |
| P95 延迟 | 2s | 500ms | 200ms |

### 生产环境 Checklist

- [ ] 建立评估数据集（Query + 相关文档）
- [ ] 实现混合检索（向量 + 关键词）
- [ ] 添加 Rerank 层
- [ ] 选择合适的向量索引
- [ ] 实现缓存层（Redis）
- [ ] 监控关键指标（延迟、QPS、错误率）
- [ ] A/B 测试对比优化效果
- [ ] 定期更新向量索引

---

## 🎓 小练习

1. **基础练习**：
   - 实现一个简单的评估器，计算 Recall@5 和 Precision@5

2. **进阶练习**：
   - 对比 Flat、IVF、HNSW 三种索引的性能和精度

3. **实战练习**：
   - 优化第 8 阶段的 Demo，提升 Recall@10 到 80% 以上

---

## 📚 推荐阅读

- [FAISS 性能调优指南](https://github.com/facebookresearch/faiss/wiki/Guidelines-for-choosing-an-index)
- [RAGAS: RAG 评估框架](https://github.com/explodinggradients/ragas)
- [混合检索最佳实践](https://www.pinecone.io/learn/hybrid-search/)
- [MRR/NDCG 详解](https://en.wikipedia.org/wiki/Information_retrieval#Evaluation_measures)

---

## 🎉 总结

恭喜你完成了 RAG 学习之旅！现在你已经掌握了：

✅ RAG 核心概念和工作原理
✅ Embedding 向量化技术
✅ 向量数据库的使用
✅ Chunk 切分策略
✅ 检索流程和优化
✅ Rerank 二次排序
✅ Prompt 设计技巧
✅ 完整的 RAG Demo
✅ 系统优化方法

**下一步建议**：
1. 用真实数据跑通 Demo
2. 根据业务场景调整参数
3. 建立持续评估机制
4. 探索更高级的技术（Multi-Agent、GraphRAG 等）
