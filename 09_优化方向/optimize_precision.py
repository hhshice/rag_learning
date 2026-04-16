# -*- coding: utf-8 -*-
"""
精度优化示例 - Rerank + 元数据过滤 + 阈值控制

核心策略：
1. Rerank：使用 Cross-Encoder 精确重排序
2. 元数据过滤：根据文档属性筛选
3. 阈值控制：只保留高置信度结果

安装依赖：pip install sentence-transformers chromadb
"""

from sentence_transformers import CrossEncoder
from typing import List, Dict, Optional
import chromadb


class PrecisionOptimizer:
    """精度优化器"""
    
    def __init__(
        self,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        初始化精度优化器
        
        Args:
            rerank_model: Cross-Encoder 模型名称
        """
        self.reranker = CrossEncoder(rerank_model)
        print(f"✅ 加载 Rerank 模型: {rerank_model}")
    
    def metadata_filter(
        self,
        results: List[Dict],
        filters: Dict
    ) -> List[Dict]:
        """
        元数据过滤：根据元数据筛选文档
        
        Args:
            results: 检索结果列表
            filters: 过滤条件，如 {'category': '技术', 'year': 2024}
        
        Returns:
            过滤后的结果
        """
        filtered = []
        for doc in results:
            match = True
            metadata = doc.get('metadata', {})
            
            for key, value in filters.items():
                # 支持列表匹配
                if isinstance(value, list):
                    if metadata.get(key) not in value:
                        match = False
                        break
                else:
                    if metadata.get(key) != value:
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
        
        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回 top_k 个结果
        
        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []
        
        # 构造 Query-Doc 对
        pairs = [(query, doc['text']) for doc in documents]
        
        # 计算相关性分数
        scores = self.reranker.predict(pairs)
        
        # 按分数排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 添加分数并返回
        results = []
        for doc, score in scored_docs[:top_k]:
            results.append({
                **doc,
                'rerank_score': float(score)
            })
        
        return results
    
    def threshold_filter(
        self,
        results: List[Dict],
        threshold: float = 0.5,
        score_key: str = 'rerank_score'
    ) -> List[Dict]:
        """
        阈值过滤：只保留分数高于阈值的文档
        
        Args:
            results: 结果列表
            threshold: 分数阈值
            score_key: 分数字段名
        
        Returns:
            过滤后的结果
        """
        return [
            doc for doc in results
            if doc.get(score_key, 0) >= threshold
        ]
    
    def optimize(
        self,
        query: str,
        candidates: List[Dict],
        filters: Optional[Dict] = None,
        top_k: int = 5,
        threshold: float = 0.3
    ) -> List[Dict]:
        """
        完整的精度优化流程
        
        流程：
        1. 元数据过滤（可选）
        2. Rerank 重排序
        3. 阈值过滤
        
        Args:
            query: 查询文本
            candidates: 候选文档列表
            filters: 元数据过滤条件
            top_k: 返回 top_k 个结果
            threshold: 置信度阈值
        
        Returns:
            优化后的结果
        """
        # Step 1: 元数据过滤
        if filters:
            candidates = self.metadata_filter(candidates, filters)
            print(f"  📌 元数据过滤后: {len(candidates)} 个文档")
        
        # Step 2: Rerank
        reranked = self.rerank(query, candidates, top_k=min(top_k * 2, len(candidates)))
        print(f"  🔄 Rerank 后: {len(reranked)} 个文档")
        
        # Step 3: 阈值过滤
        filtered = self.threshold_filter(reranked, threshold)
        print(f"  ✅ 阈值过滤后 (>{threshold}): {len(filtered)} 个文档")
        
        return filtered


class RAGSystem:
    """简化的 RAG 系统"""
    
    def __init__(self):
        """初始化 RAG 系统"""
        self.client = chromadb.PersistentClient(path="./chroma_db_precision")
        self.collection = self.client.get_or_create_collection(
            name="precision_demo",
            metadata={"description": "精度优化演示"}
        )
        self.optimizer = PrecisionOptimizer()
    
    def add_documents(self, documents: List[Dict]):
        """
        添加文档到知识库
        
        Args:
            documents: 文档列表，每个文档包含 text, id, metadata
        """
        texts = [doc['text'] for doc in documents]
        ids = [doc['id'] for doc in documents]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"✅ 已添加 {len(documents)} 个文档到知识库")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict] = None,
        rerank_top_k: int = 5,
        threshold: float = 0.3
    ) -> List[Dict]:
        """
        检索流程：召回 -> 精排 -> 过滤
        
        Args:
            query: 查询文本
            top_k: 召回数量
            filters: 元数据过滤条件
            rerank_top_k: Rerank 后返回数量
            threshold: 置信度阈值
        
        Returns:
            最终结果
        """
        print(f"\n🔍 查询: {query}")
        print("-" * 60)
        
        # Step 1: 向量召回
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        candidates = []
        for i in range(len(results['ids'][0])):
            candidates.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results.get('metadatas') else {}
            })
        
        print(f"  📥 向量召回: {len(candidates)} 个文档")
        
        # Step 2: 精度优化
        final_results = self.optimizer.optimize(
            query=query,
            candidates=candidates,
            filters=filters,
            top_k=rerank_top_k,
            threshold=threshold
        )
        
        return final_results


def main():
    print("=" * 60)
    print("🎯 精度优化演示：Rerank + 过滤 + 阈值")
    print("=" * 60)
    
    # 初始化系统
    rag = RAGSystem()
    
    # 准备示例文档（带元数据）
    documents = [
        {
            'id': 'doc1',
            'text': 'RAG（检索增强生成）是一种结合检索和生成的 AI 技术。',
            'metadata': {'category': '技术', 'year': 2024, 'source': 'blog'}
        },
        {
            'id': 'doc2',
            'text': '向量数据库用于存储和检索文本的向量表示。',
            'metadata': {'category': '技术', 'year': 2023, 'source': 'docs'}
        },
        {
            'id': 'doc3',
            'text': 'Python 是一门流行的编程语言，广泛用于数据科学。',
            'metadata': {'category': '编程', 'year': 2024, 'source': 'tutorial'}
        },
        {
            'id': 'doc4',
            'text': '深度学习是机器学习的一个分支，使用神经网络进行特征学习。',
            'metadata': {'category': '技术', 'year': 2024, 'source': 'paper'}
        },
        {
            'id': 'doc5',
            'text': 'RAG 系统的核心组件包括：检索器、增强器、生成器。',
            'metadata': {'category': '技术', 'year': 2024, 'source': 'blog'}
        },
        {
            'id': 'doc6',
            'text': 'Embedding 模型将文本转换为高维向量。',
            'metadata': {'category': '技术', 'year': 2023, 'source': 'docs'}
        },
        {
            'id': 'doc7',
            'text': 'FastAPI 是一个现代的 Python Web 框架。',
            'metadata': {'category': '编程', 'year': 2024, 'source': 'tutorial'}
        },
        {
            'id': 'doc8',
            'text': 'RAG 的优化方向包括召回率、精度和速度。',
            'metadata': {'category': '技术', 'year': 2024, 'source': 'blog'}
        }
    ]
    
    # 清空并重新添加
    try:
        rag.client.delete_collection("precision_demo")
        rag.collection = rag.client.get_or_create_collection(
            name="precision_demo"
        )
    except:
        pass
    
    rag.add_documents(documents)
    
    # 测试不同优化策略
    print("\n" + "=" * 60)
    print("📊 优化策略对比")
    print("=" * 60)
    
    # 测试 1: 无优化（仅向量检索）
    print("\n【测试 1】仅向量检索（无优化）")
    print("-" * 60)
    results = rag.collection.query(
        query_texts=["RAG 技术"],
        n_results=5
    )
    
    for i in range(len(results['ids'][0])):
        print(f"{i+1}. {results['documents'][0][i][:50]}...")
    
    # 测试 2: 元数据过滤
    print("\n【测试 2】元数据过滤（category=技术, year=2024）")
    results = rag.retrieve(
        query="RAG 技术",
        top_k=10,
        filters={'category': '技术', 'year': 2024},
        rerank_top_k=5,
        threshold=0.0
    )
    
    print("\n最终结果:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. [{doc['id']}] 分数: {doc.get('rerank_score', 0):.3f}")
        print(f"   文本: {doc['text'][:50]}...")
        print(f"   元数据: {doc['metadata']}")
    
    # 测试 3: 完整优化
    print("\n【测试 3】完整优化（Rerank + 阈值过滤）")
    results = rag.retrieve(
        query="什么是 RAG",
        top_k=10,
        rerank_top_k=5,
        threshold=0.5
    )
    
    print("\n最终结果（分数 > 0.5）:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. [{doc['id']}] 分数: {doc.get('rerank_score', 0):.3f}")
        print(f"   文本: {doc['text']}")
    
    print("\n" + "=" * 60)
    print("✅ 精度优化演示完成")
    print("=" * 60)
    print("\n关键点:")
    print("1. Rerank 显著提升排序精度")
    print("2. 元数据过滤减少无关文档")
    print("3. 阈值控制确保结果质量")
    print("4. 三者结合效果最佳")


if __name__ == "__main__":
    main()
