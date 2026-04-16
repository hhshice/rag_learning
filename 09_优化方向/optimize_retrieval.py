# -*- coding: utf-8 -*-
"""
召回率优化示例 - 混合检索 + Query 扩展

核心策略：
1. 混合检索：融合向量检索和关键词检索
2. Query 扩展：同义词扩展提高召回
3. 多路召回：从不同角度召回文档

安装依赖：pip install chromadb
"""

import chromadb
from typing import List, Dict, Set
import re


class HybridSearch:
    """混合检索器：融合向量检索和关键词检索"""
    
    def __init__(self, alpha: float = 0.7):
        """
        初始化混合检索器
        
        Args:
            alpha: 向量检索权重 (0-1)，(1-alpha) 为关键词检索权重
        """
        self.alpha = alpha
        self.client = chromadb.PersistentClient(path="./chroma_db_hybrid")
        self.collection = self.client.get_or_create_collection(
            name="hybrid_demo",
            metadata={"description": "混合检索演示"}
        )
    
    def add_documents(self, documents: List[str], ids: List[str]):
        """添加文档到集合"""
        self.collection.upsert(
            ids=ids,
            documents=documents
        )
        print(f"✅ 已添加 {len(documents)} 个文档")
    
    def vector_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """向量检索"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        docs = []
        for i in range(len(results['ids'][0])):
            docs.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'distance': results['distances'][0][i] if results.get('distances') else 0,
                'type': 'vector'
            })
        
        return docs
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        关键词检索（简化版，实际应使用 BM25）
        这里用 Chroma 的 where_document 过滤
        """
        # 简化实现：提取关键词并在文档中查找
        keywords = self._extract_keywords(query)
        
        # 获取所有文档（实际应使用倒排索引）
        all_docs = self.collection.get()
        
        # 计算关键词匹配分数
        scored_docs = []
        for i, doc_text in enumerate(all_docs['documents']):
            score = sum(
                1 for kw in keywords
                if kw.lower() in doc_text.lower()
            )
            if score > 0:
                scored_docs.append({
                    'id': all_docs['ids'][i],
                    'text': doc_text,
                    'score': score,
                    'type': 'keyword'
                })
        
        # 按分数排序
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_docs[:top_k]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（简化版）"""
        # 移除停用词（简化）
        stop_words = {'的', '是', '在', '和', '了', '有', '我', '他', '她'}
        words = re.findall(r'[\w]+', text)
        return [w for w in words if w not in stop_words and len(w) > 1]
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        混合检索：融合向量和关键词结果
        
        策略：
        1. 向量检索召回 top_k * 2 个文档
        2. 关键词检索召回 top_k * 2 个文档
        3. 计算混合分数并排序
        """
        # 多召回一些文档
        vector_results = self.vector_search(query, top_k=top_k * 2)
        keyword_results = self.keyword_search(query, top_k=top_k * 2)
        
        # 合并结果
        all_docs = {}
        
        # 向量检索结果归一化
        max_dist = max([d['distance'] for d in vector_results], default=1)
        for i, doc in enumerate(vector_results):
            doc_id = doc['id']
            # 距离越小越好，转换为分数
            score = 1 - (doc['distance'] / max_dist if max_dist > 0 else 0)
            rank_score = 1 / (i + 1)
            
            all_docs[doc_id] = {
                **doc,
                'vector_score': score * rank_score,
                'keyword_score': 0
            }
        
        # 关键词检索结果归一化
        max_score = max([d['score'] for d in keyword_results], default=1)
        for i, doc in enumerate(keyword_results):
            doc_id = doc['id']
            score = doc['score'] / max_score if max_score > 0 else 0
            rank_score = 1 / (i + 1)
            
            if doc_id in all_docs:
                all_docs[doc_id]['keyword_score'] = score * rank_score
            else:
                all_docs[doc_id] = {
                    **doc,
                    'vector_score': 0,
                    'keyword_score': score * rank_score
                }
        
        # 计算混合分数
        for doc_id in all_docs:
            all_docs[doc_id]['final_score'] = (
                self.alpha * all_docs[doc_id]['vector_score'] +
                (1 - self.alpha) * all_docs[doc_id]['keyword_score']
            )
        
        # 按混合分数排序
        sorted_results = sorted(
            all_docs.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )
        
        return sorted_results[:top_k]


class QueryExpander:
    """Query 扩展器"""
    
    def __init__(self):
        # 同义词词典（示例）
        self.synonyms = {
            'RAG': ['检索增强生成', 'Retrieval-Augmented Generation', '检索增强'],
            '向量': ['Vector', 'Embedding', '嵌入'],
            '数据库': ['Database', 'DB', '存储'],
            '模型': ['Model', 'AI', '人工智能'],
            '搜索': ['Search', '检索', '查找'],
            '优化': ['Optimization', '改进', '提升']
        }
    
    def expand(self, query: str) -> List[str]:
        """
        扩展 Query：生成多个相关 Query
        
        Args:
            query: 原始查询
        
        Returns:
            扩展后的查询列表
        """
        queries = [query]
        
        # 同义词扩展
        for word, synonyms in self.synonyms.items():
            if word in query:
                for synonym in synonyms:
                    expanded = query.replace(word, synonym)
                    queries.append(expanded)
        
        return list(set(queries))  # 去重


def main():
    print("=" * 60)
    print("🔍 召回率优化演示：混合检索 + Query 扩展")
    print("=" * 60)
    
    # 初始化
    searcher = HybridSearch(alpha=0.7)
    expander = QueryExpander()
    
    # 添加示例文档
    documents = [
        "RAG（检索增强生成）是一种结合检索和生成的技术，让大模型能够基于外部知识回答问题。",
        "向量数据库是 RAG 系统的核心组件，用于存储和检索文本的向量表示。",
        "Embedding 模型将文本转换为向量，是连接文本和向量数据库的桥梁。",
        "RAG 系统包含三个核心步骤：检索、增强、生成。",
        "向量检索通过计算向量相似度来找到最相关的文档。",
        "关键词搜索是一种传统的检索方式，通过匹配关键词来查找文档。",
        "混合检索结合了向量检索和关键词检索的优点，召回率更高。",
        "Query 扩展通过同义词替换等方式扩展查询，提高召回率。",
        "召回率是衡量检索系统效果的重要指标，表示召回了多少相关文档。",
        "优化召回率的方法包括：混合检索、Query 扩展、增加召回数量等。"
    ]
    
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    # 清空并重新添加
    try:
        searcher.client.delete_collection("hybrid_demo")
        searcher.collection = searcher.client.get_or_create_collection(
            name="hybrid_demo"
        )
    except:
        pass
    
    searcher.add_documents(documents, ids)
    
    # 测试查询
    query = "RAG 系统优化"
    
    print("\n" + "=" * 60)
    print(f"📝 原始查询: {query}")
    print("=" * 60)
    
    # Query 扩展
    expanded_queries = expander.expand(query)
    print(f"\n🔄 扩展后的查询 ({len(expanded_queries)} 个):")
    for i, q in enumerate(expanded_queries, 1):
        print(f"  {i}. {q}")
    
    # 单独检索对比
    print("\n" + "-" * 60)
    print("📊 检索方式对比")
    print("-" * 60)
    
    # 向量检索
    print("\n1️⃣ 向量检索结果:")
    vector_results = searcher.vector_search(query, top_k=5)
    for i, doc in enumerate(vector_results, 1):
        print(f"  {i}. [{doc['id']}] {doc['text'][:50]}...")
    
    # 关键词检索
    print("\n2️⃣ 关键词检索结果:")
    keyword_results = searcher.keyword_search(query, top_k=5)
    for i, doc in enumerate(keyword_results, 1):
        print(f"  {i}. [{doc['id']}] {doc['text'][:50]}...")
    
    # 混合检索
    print("\n3️⃣ 混合检索结果:")
    hybrid_results = searcher.hybrid_search(query, top_k=5)
    for i, doc in enumerate(hybrid_results, 1):
        print(f"  {i}. [{doc['id']}] 分数: {doc['final_score']:.3f}")
        print(f"     {doc['text'][:50]}...")
    
    # 多 Query 召回
    print("\n" + "-" * 60)
    print("🔄 多 Query 召回策略")
    print("-" * 60)
    
    all_retrieved_ids = set()
    for q in expanded_queries:
        results = searcher.hybrid_search(q, top_k=5)
        for doc in results:
            all_retrieved_ids.add(doc['id'])
    
    print(f"\n原始查询召回: {len(vector_results)} 个文档")
    print(f"多 Query 召回: {len(all_retrieved_ids)} 个文档")
    print(f"召回提升: {len(all_retrieved_ids) - len(vector_results)} 个文档")
    
    print("\n" + "=" * 60)
    print("✅ 召回率优化演示完成")
    print("=" * 60)
    print("\n关键点:")
    print("1. 混合检索融合向量+关键词，召回更全面")
    print("2. Query 扩展增加查询多样性，提高召回")
    print("3. 调整 alpha 参数平衡向量和关键词权重")


if __name__ == "__main__":
    main()
