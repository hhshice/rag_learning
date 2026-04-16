# -*- coding: utf-8 -*-
"""
最小 RAG 系统核心实现
整合了文档切分、向量检索、Rerank、Prompt 构建等所有核心功能

安装依赖：
pip install chromadb sentence-transformers langchain-text-splitters
"""

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
import os


class MiniRAG:
    """
    最小 RAG 系统
    
    功能：
    - 文档加载与切分
    - 向量化与存储
    - 语义检索
    - Rerank 精排
    - Prompt 构建
    """
    
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "knowledge_base",
        embedding_model: str = "all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 50
    ):
        """
        初始化 RAG 系统
        
        Args:
            persist_dir: 向量数据库持久化目录
            collection_name: 集合名称
            embedding_model: Embedding 模型名称
            rerank_model: Rerank 模型名称
            chunk_size: 文档切分大小
            chunk_overlap: 文档切分重叠大小
        """
        print("=" * 60)
        print("🚀 初始化 RAG 系统")
        print("=" * 60)
        
        # 初始化向量数据库
        print(f"✅ 初始化向量数据库: {persist_dir}")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG Demo 知识库"}
        )
        
        # 加载 Embedding 模型
        print(f"✅ 加载 Embedding 模型: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # 加载 Rerank 模型
        print(f"✅ 加载 Rerank 模型: {rerank_model}")
        self.rerank_model = CrossEncoder(rerank_model)
        
        # 文档切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        
        print("=" * 60)
        print("✅ RAG 系统初始化完成")
        print(f"   知识库文档数：{self.collection.count()}")
        print("=" * 60)
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None
    ):
        """
        添加文档到知识库
        
        流程：
        1. 切分文档
        2. 生成 Chunk ID
        3. 存储到向量数据库
        
        Args:
            documents: 文档列表
            metadatas: 元数据列表（可选）
        """
        print("\n" + "=" * 60)
        print(f"📄 添加文档到知识库")
        print("=" * 60)
        print(f"文档数量：{len(documents)}")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for i, doc in enumerate(documents):
            # 切分文档
            chunks = self.text_splitter.split_text(doc)
            
            print(f"  文档 {i+1}: 切分为 {len(chunks)} 个 Chunk")
            
            for j, chunk in enumerate(chunks):
                chunk_id = f"doc_{i}_chunk_{j}"
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                
                # 元数据
                if metadatas and i < len(metadatas):
                    meta = {**metadatas[i], "chunk_index": j, "total_chunks": len(chunks)}
                else:
                    meta = {"doc_index": i, "chunk_index": j, "total_chunks": len(chunks)}
                all_metadatas.append(meta)
        
        # 添加到向量数据库
        self.collection.upsert(
            ids=all_ids,
            documents=all_chunks,
            metadatas=all_metadatas
        )
        
        print("\n" + "=" * 60)
        print("✅ 文档添加完成")
        print(f"   新增 Chunk 数：{len(all_chunks)}")
        print(f"   知识库总文档数：{self.collection.count()}")
        print("=" * 60)
    
    def add_document_from_file(
        self,
        file_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        从文件添加文档
        
        Args:
            file_path: 文件路径
            metadata: 元数据（可选）
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.add_documents([content], [metadata] if metadata else None)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True,
        rerank_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        检索相关文档
        
        流程：
        1. 向量检索（召回）
        2. Rerank 精排（可选）
        3. 阈值过滤
        
        Args:
            query: 查询文本
            top_k: 返回文档数量
            use_rerank: 是否使用 Rerank
            rerank_threshold: Rerank 分数阈值
        
        Returns:
            results: 检索结果列表
        """
        # 向量检索（召回）
        retrieve_k = min(top_k * 3, self.collection.count())
        
        search_results = self.collection.query(
            query_texts=[query],
            n_results=max(retrieve_k, 1)
        )
        
        documents = search_results['documents'][0]
        ids = search_results['ids'][0]
        distances = search_results['distances'][0]
        metadatas = search_results['metadatas'][0]
        
        # Rerank 精排
        if use_rerank and len(documents) > 0:
            pairs = [[query, doc] for doc in documents]
            scores = self.rerank_model.predict(pairs)
            
            # 组合并排序
            results = []
            for i, (doc_id, doc, dist, meta, score) in enumerate(
                zip(ids, documents, distances, metadatas, scores)
            ):
                results.append({
                    "id": doc_id,
                    "content": doc,
                    "vector_similarity": 1 - dist,
                    "rerank_score": float(score),
                    "metadata": meta
                })
            
            # 按 rerank 分数排序
            results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # 阈值过滤
            if rerank_threshold > 0:
                results = [r for r in results if r['rerank_score'] >= rerank_threshold]
            
            results = results[:top_k]
        
        else:
            results = []
            for doc_id, doc, dist, meta in zip(ids, documents, distances, metadatas):
                results.append({
                    "id": doc_id,
                    "content": doc,
                    "vector_similarity": 1 - dist,
                    "metadata": meta
                })
            results = results[:top_k]
        
        return results
    
    def build_prompt(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        style: str = "balanced"
    ) -> str:
        """
        构建 Prompt
        
        Args:
            query: 用户问题
            documents: 检索结果
            style: 风格
                - strict: 严格约束，只使用文档信息
                - balanced: 平衡模式，允许适当推理
                - creative: 创意模式，允许更多发挥
        
        Returns:
            prompt: 构建好的 Prompt
        """
        # 格式化文档
        docs_text = "\n\n".join([
            f"【参考资料{i+1}】\n来源：{doc['metadata'].get('source', '未知')}\n{doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        if style == "strict":
            return f"""你是一个专业的问答助手。

【重要规则】
1. 只使用参考资料中的信息回答问题
2. 如果参考资料中没有相关信息，请明确说"参考资料中未找到相关信息"
3. 不要编造、推测或使用外部知识
4. 回答要简洁准确

参考资料：
{docs_text}

用户问题：{query}

请根据规则回答："""
        
        elif style == "balanced":
            return f"""你是一个专业的问答助手。

请根据参考资料回答用户问题。如果参考资料信息不足，可以结合常识进行适当补充，但要明确标注。

参考资料：
{docs_text}

用户问题：{query}

请回答："""
        
        else:  # creative
            return f"""你是一个专业的问答助手。

请根据参考资料回答用户问题。可以结合参考资料和你的知识，给出全面详细的回答。

参考资料：
{docs_text}

用户问题：{query}

请详细回答："""
    
    def query(
        self,
        question: str,
        top_k: int = 3,
        use_rerank: bool = True,
        rerank_threshold: float = 0.0,
        prompt_style: str = "balanced",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        完整查询流程
        
        流程：
        1. 检索相关文档
        2. 构建 Prompt
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
            use_rerank: 是否使用 Rerank
            rerank_threshold: Rerank 分数阈值
            prompt_style: Prompt 风格
            verbose: 是否显示详细信息
        
        Returns:
            result: 包含 prompt 和检索结果的字典
        """
        if verbose:
            print("\n" + "=" * 60)
            print(f"🔍 查询：{question}")
            print("=" * 60)
        
        # 1. 检索
        documents = self.retrieve(
            question,
            top_k=top_k,
            use_rerank=use_rerank,
            rerank_threshold=rerank_threshold
        )
        
        if verbose:
            print(f"\n📚 检索到 {len(documents)} 个相关文档：")
            print("-" * 60)
            
            for i, doc in enumerate(documents, 1):
                print(f"  [{i}] 来源：{doc['metadata'].get('source', '未知')}")
                print(f"      向量相似度：{doc['vector_similarity']:.4f}")
                if 'rerank_score' in doc:
                    print(f"      Rerank分数：{doc['rerank_score']:.4f}")
                print(f"      内容：{doc['content'][:60]}...")
        
        # 2. 构建 Prompt
        prompt = self.build_prompt(question, documents, style=prompt_style)
        
        if verbose:
            print("\n📝 构建的 Prompt：")
            print("-" * 60)
            print(prompt)
        
        return {
            "question": question,
            "documents": documents,
            "prompt": prompt
        }
    
    def clear_collection(self):
        """清空知识库"""
        # 获取所有文档 ID
        all_ids = self.collection.get()['ids']
        
        if all_ids:
            self.collection.delete(ids=all_ids)
            print(f"✅ 已清空知识库，删除了 {len(all_ids)} 个文档")
        else:
            print("ℹ️  知识库已为空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name,
            "metadata": self.collection.metadata
        }


if __name__ == "__main__":
    # 测试代码
    print("MiniRAG 核心模块测试")
    
    # 创建 RAG 实例
    rag = MiniRAG(
        persist_dir="./test_chroma_db",
        collection_name="test_kb"
    )
    
    # 添加测试文档
    test_docs = [
        "RAG 是检索增强生成技术。",
        "向量数据库用于存储向量。"
    ]
    
    rag.add_documents(test_docs)
    
    # 测试查询
    result = rag.query("什么是 RAG？")
    
    print("\n✅ 测试完成")
