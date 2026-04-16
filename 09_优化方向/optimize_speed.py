# -*- coding: utf-8 -*-
"""
速度优化示例 - 索引优化 + 批处理 + 缓存

核心策略：
1. 向量索引优化：选择合适的索引结构（Flat/IVF/HNSW）
2. 批处理：批量 Embedding 和检索
3. 缓存：缓存热门 Query 结果

安装依赖：pip install faiss-cpu numpy chromadb redis
"""

import time
import faiss
import numpy as np
from typing import List, Tuple, Optional
import chromadb


class SpeedOptimizer:
    """速度优化器 - FAISS 索引"""
    
    def __init__(self, embedding_dim: int = 128):
        """
        初始化速度优化器
        
        Args:
            embedding_dim: 向量维度
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.index_type = None
    
    # ========== 索引构建 ==========
    
    def build_flat_index(self, embeddings: np.ndarray) -> float:
        """
        构建 Flat 索引（暴力搜索）
        
        特点：
        - 精度最高
        - 速度最慢
        - 适合小规模数据（< 10万）
        
        Args:
            embeddings: 向量矩阵 [N, D]
        
        Returns:
            构建时间（秒）
        """
        start = time.time()
        
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        self.index_type = "Flat"
        
        build_time = time.time() - start
        return build_time
    
    def build_ivf_index(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 100
    ) -> float:
        """
        构建 IVF 索引（倒排索引）
        
        特点：
        - 速度快
        - 精度略有下降
        - 适合中大规模数据（10万 - 1000万）
        
        Args:
            embeddings: 向量矩阵 [N, D]
            n_clusters: 聚类中心数量
        
        Returns:
            构建时间（秒）
        """
        start = time.time()
        
        # 量化器
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        
        # IVF 索引
        self.index = faiss.IndexIVFFlat(
            quantizer,
            self.embedding_dim,
            n_clusters
        )
        
        # 训练
        self.index.train(embeddings.astype('float32'))
        self.index.add(embeddings.astype('float32'))
        self.index_type = "IVF"
        
        build_time = time.time() - start
        return build_time
    
    def build_hnsw_index(
        self,
        embeddings: np.ndarray,
        m: int = 32,
        ef_search: int = 64
    ) -> float:
        """
        构建 HNSW 索引（图索引）
        
        特点：
        - 速度极快
        - 精度高
        - 内存占用大
        - 适合大规模数据（> 100万）
        
        Args:
            embeddings: 向量矩阵 [N, D]
            m: 图连接数
            ef_search: 搜索时访问的节点数
        
        Returns:
            构建时间（秒）
        """
        start = time.time()
        
        self.index = faiss.IndexHNSWFlat(
            self.embedding_dim,
            m,
            faiss.METRIC_INNER_PRODUCT
        )
        self.index.hnsw.efSearch = ef_search
        self.index.add(embeddings.astype('float32'))
        self.index_type = "HNSW"
        
        build_time = time.time() - start
        return build_time
    
    def search(
        self,
        query: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        向量检索
        
        Args:
            query: 查询向量 [D] 或 [N, D]
            k: 返回 top_k 个结果
        
        Returns:
            distances: 距离数组
            indices: 索引数组
            search_time: 检索时间（毫秒）
        """
        # 确保维度正确
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = query.astype('float32')
        
        start = time.time()
        distances, indices = self.index.search(query, k)
        search_time = (time.time() - start) * 1000  # 转换为毫秒
        
        return distances[0], indices[0], search_time
    
    def batch_search(
        self,
        queries: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        批量检索
        
        Args:
            queries: 查询向量矩阵 [N, D]
            k: 返回 top_k 个结果
        
        Returns:
            distances: 距离矩阵
            indices: 索引矩阵
            search_time: 检索时间（毫秒）
        """
        queries = queries.astype('float32')
        
        start = time.time()
        distances, indices = self.index.search(queries, k)
        search_time = (time.time() - start) * 1000
        
        return distances, indices, search_time


class CacheManager:
    """简单的缓存管理器（模拟 Redis）"""
    
    def __init__(self, max_size: int = 1000):
        """
        初始化缓存管理器
        
        Args:
            max_size: 最大缓存数量
        """
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get(self, key: str) -> Optional[List]:
        """获取缓存"""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def set(self, key: str, value: List):
        """设置缓存"""
        # LRU 淘汰
        if len(self.cache) >= self.max_size:
            # 移除访问次数最少的
            min_key = min(self.access_count, key=self.access_count.get)
            del self.cache[min_key]
            del self.access_count[min_key]
        
        self.cache[key] = value
        self.access_count[key] = 1
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_count.clear()
    
    def stats(self) -> dict:
        """缓存统计"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': 0.0  # 需要额外跟踪
        }


def compare_indexes():
    """对比不同索引的性能"""
    print("=" * 70)
    print("📊 FAISS 索引性能对比测试")
    print("=" * 70)
    
    # 生成模拟数据
    n_docs = 10000
    embedding_dim = 128
    
    print(f"\n数据规模: {n_docs} 个文档, {embedding_dim} 维向量\n")
    
    # 生成随机向量（模拟 Embedding）
    np.random.seed(42)
    embeddings = np.random.randn(n_docs, embedding_dim).astype('float32')
    
    # 归一化（用于内积相似度）
    faiss.normalize_L2(embeddings)
    
    optimizer = SpeedOptimizer(embedding_dim=embedding_dim)
    
    # 测试索引
    results = []
    
    # ========== Flat 索引 ==========
    print("1️⃣ Flat 索引（暴力搜索）")
    print("-" * 70)
    
    build_time = optimizer.build_flat_index(embeddings)
    print(f"  构建时间: {build_time:.3f}s")
    
    # 单次检索
    query = np.random.randn(embedding_dim).astype('float32')
    faiss.normalize_L2(query.reshape(1, -1))
    
    distances, indices, search_time = optimizer.search(query, k=10)
    print(f"  单次检索: {search_time:.2f}ms")
    
    # 批量检索
    n_queries = 100
    queries = np.random.randn(n_queries, embedding_dim).astype('float32')
    faiss.normalize_L2(queries)
    
    distances, indices, batch_time = optimizer.batch_search(queries, k=10)
    print(f"  批量检索 ({n_queries} 个): {batch_time:.2f}ms")
    print(f"  平均每个: {batch_time/n_queries:.2f}ms")
    
    results.append({
        'index': 'Flat',
        'build_time': build_time,
        'search_time': search_time,
        'batch_time': batch_time
    })
    
    # ========== IVF 索引 ==========
    print("\n2️⃣ IVF 索引（倒排索引）")
    print("-" * 70)
    
    build_time = optimizer.build_ivf_index(embeddings, n_clusters=100)
    print(f"  构建时间: {build_time:.3f}s")
    
    distances, indices, search_time = optimizer.search(query, k=10)
    print(f"  单次检索: {search_time:.2f}ms")
    
    distances, indices, batch_time = optimizer.batch_search(queries, k=10)
    print(f"  批量检索 ({n_queries} 个): {batch_time:.2f}ms")
    print(f"  平均每个: {batch_time/n_queries:.2f}ms")
    
    results.append({
        'index': 'IVF',
        'build_time': build_time,
        'search_time': search_time,
        'batch_time': batch_time
    })
    
    # ========== HNSW 索引 ==========
    print("\n3️⃣ HNSW 索引（图索引）")
    print("-" * 70)
    
    build_time = optimizer.build_hnsw_index(embeddings, m=32, ef_search=64)
    print(f"  构建时间: {build_time:.3f}s")
    
    distances, indices, search_time = optimizer.search(query, k=10)
    print(f"  单次检索: {search_time:.2f}ms")
    
    distances, indices, batch_time = optimizer.batch_search(queries, k=10)
    print(f"  批量检索 ({n_queries} 个): {batch_time:.2f}ms")
    print(f"  平均每个: {batch_time/n_queries:.2f}ms")
    
    results.append({
        'index': 'HNSW',
        'build_time': build_time,
        'search_time': search_time,
        'batch_time': batch_time
    })
    
    # ========== 总结 ==========
    print("\n" + "=" * 70)
    print("📈 性能对比总结")
    print("=" * 70)
    
    print(f"\n{'索引类型':<10} {'构建时间':<15} {'单次检索':<15} {'批量检索':<15}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['index']:<10} {r['build_time']:.3f}s{'':<10} "
              f"{r['search_time']:.2f}ms{'':<10} {r['batch_time']:.2f}ms")
    
    print("\n💡 建议:")
    print("  - < 10万数据: 使用 Flat 索引")
    print("  - 10万-1000万: 使用 IVF 索引")
    print("  - > 1000万: 使用 HNSW 索引")


def demo_caching():
    """演示缓存效果"""
    print("\n" + "=" * 70)
    print("💾 缓存优化演示")
    print("=" * 70)
    
    # 初始化
    cache = CacheManager(max_size=100)
    optimizer = SpeedOptimizer(embedding_dim=128)
    
    # 准备数据
    n_docs = 5000
    embeddings = np.random.randn(n_docs, 128).astype('float32')
    faiss.normalize_L2(embeddings)
    
    optimizer.build_hnsw_index(embeddings)
    
    # 模拟热门查询
    hot_queries = [
        "什么是 RAG",
        "向量数据库",
        "Embedding 技术"
    ]
    
    # 模拟查询向量
    query_vectors = {
        "什么是 RAG": np.random.randn(128).astype('float32'),
        "向量数据库": np.random.randn(128).astype('float32'),
        "Embedding 技术": np.random.randn(128).astype('float32')
    }
    
    # 归一化
    for key in query_vectors:
        faiss.normalize_L2(query_vectors[key].reshape(1, -1))
    
    print("\n模拟 100 次查询（含重复热门查询）\n")
    
    # 无缓存
    print("【无缓存】")
    start = time.time()
    
    for i in range(100):
        # 70% 概率查询热门问题
        if np.random.random() < 0.7:
            query = np.random.choice(hot_queries)
        else:
            query = f"随机查询 {i}"
        
        # 直接检索
        if query in query_vectors:
            optimizer.search(query_vectors[query], k=5)
    
    no_cache_time = (time.time() - start) * 1000
    print(f"  总耗时: {no_cache_time:.2f}ms")
    
    # 有缓存
    print("\n【有缓存】")
    cache.clear()
    start = time.time()
    
    for i in range(100):
        # 70% 概率查询热门问题
        if np.random.random() < 0.7:
            query = np.random.choice(hot_queries)
        else:
            query = f"随机查询 {i}"
        
        # 检查缓存
        cached = cache.get(query)
        if cached:
            continue  # 缓存命中
        
        # 未命中，执行检索并缓存
        if query in query_vectors:
            distances, indices, _ = optimizer.search(query_vectors[query], k=5)
            cache.set(query, list(zip(distances, indices)))
    
    with_cache_time = (time.time() - start) * 1000
    print(f"  总耗时: {with_cache_time:.2f}ms")
    print(f"  缓存命中率: {len(cache.cache) / 100 * 100:.1f}%")
    print(f"  性能提升: {(no_cache_time - with_cache_time) / no_cache_time * 100:.1f}%")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("🚀 RAG 速度优化演示")
    print("=" * 70)
    
    # 1. 索引对比
    compare_indexes()
    
    # 2. 缓存演示
    demo_caching()
    
    print("\n" + "=" * 70)
    print("✅ 速度优化演示完成")
    print("=" * 70)
    
    print("\n🎯 优化要点:")
    print("1. 选择合适的索引（根据数据规模）")
    print("2. 使用批量检索提升吞吐量")
    print("3. 添加缓存层减少重复计算")
    print("4. 监控 P50/P95/P99 延迟")


if __name__ == "__main__":
    main()
