# -*- coding: utf-8 -*-
"""
完整流程示例：文档切分 → 向量化 → 存储到向量数据库
演示 RAG 系统中 Chunk 的完整处理流程

安装依赖：pip install chromadb langchain-text-splitters
"""

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_sample_documents():
    """创建示例文档"""
    documents = [
        {
            "id": "python_intro",
            "content": """
# Python 编程入门

Python 是一门流行的编程语言。它的语法简洁，易于学习。
Python 广泛应用于 Web 开发、数据科学、人工智能等领域。

## 变量与数据类型

Python 支持多种数据类型：整数、浮点数、字符串、列表、字典等。
变量无需声明类型，Python 会自动推断。

## 控制流程

Python 提供 if-elif-else 条件判断语句。
循环语句包括 for 循环和 while 循环。
""",
            "metadata": {"category": "编程", "author": "张三"}
        },
        {
            "id": "rag_intro",
            "content": """
# RAG 技术详解

RAG 是检索增强生成的缩写。它让大模型能够通过检索外部知识来增强回答。
RAG 的核心组件包括：文档切分、向量数据库、Embedding 模型、检索算法。

## 工作流程

1. 文档加载：读取各种格式的文档
2. 文档切分：将长文档切成小块
3. 向量化：使用 Embedding 模型转换为向量
4. 存储：将向量存入向量数据库
5. 检索：根据查询向量检索相关文档
6. 生成：将检索结果喂给大模型生成回答

## 应用场景

RAG 适用于企业知识库、智能客服、文档问答等场景。
""",
            "metadata": {"category": "AI技术", "author": "李四"}
        },
        {
            "id": "vector_db",
            "content": """
# 向量数据库介绍

向量数据库是专门用于存储和检索向量的数据库。
它支持高效的相似度搜索，是 RAG 系统的核心组件。

## 常见向量数据库

- Chroma：轻量级，易于上手
- FAISS：Facebook 开源，性能强大
- Milvus：企业级，支持分布式
- Pinecone：云服务，免运维

## 使用场景

向量数据库广泛应用于推荐系统、搜索引擎、问答系统等领域。
""",
            "metadata": {"category": "数据库", "author": "王五"}
        }
    ]
    return documents


def chunk_document(text, doc_id, metadata=None):
    """
    切分单个文档
    
    Args:
        text: 文档文本
        doc_id: 文档 ID
        metadata: 文档元数据
    
    Returns:
        chunks_data: 包含 id、content、metadata 的字典列表
    """
    # 创建切分器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,       # 每个 Chunk 最大 200 字符
        chunk_overlap=40,     # 重叠 40 字符（约 20%）
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    
    # 执行切分
    chunks = splitter.split_text(text)
    
    # 构建 Chunk 数据
    chunks_data = []
    for i, chunk_content in enumerate(chunks):
        chunk_data = {
            "id": f"{doc_id}_chunk_{i}",
            "content": chunk_content,
            "metadata": {
                **(metadata or {}),
                "source_doc": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
        }
        chunks_data.append(chunk_data)
    
    return chunks_data


def store_chunks_to_chroma(chunks_data, collection):
    """
    将 Chunk 存储到 Chroma 向量数据库
    
    Args:
        chunks_data: Chunk 数据列表
        collection: Chroma 集合对象
    """
    # 准备数据
    ids = [chunk["id"] for chunk in chunks_data]
    documents = [chunk["content"] for chunk in chunks_data]
    metadatas = [chunk["metadata"] for chunk in chunks_data]
    
    # 存储到向量数据库
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )


def search_chunks(collection, query, n_results=3):
    """
    在向量数据库中检索相关 Chunk
    
    Args:
        collection: Chroma 集合对象
        query: 查询文本
        n_results: 返回结果数量
    
    Returns:
        results: 检索结果
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results


def display_search_results(query, results):
    """显示检索结果"""
    print(f"\n🔍 查询：{query}")
    print("=" * 60)
    
    for i, (doc_id, document, distance, metadata) in enumerate(
        zip(
            results['ids'][0],
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0]
        ), 1
    ):
        similarity = 1 - distance
        print(f"\n【结果 {i}】")
        print(f"相似度: {similarity:.4f}")
        print(f"来源文档: {metadata.get('source_doc', 'N/A')}")
        print(f"Chunk 索引: {metadata.get('chunk_index', 'N/A')}")
        print(f"类别: {metadata.get('category', 'N/A')}")
        print(f"内容: {document[:100]}...")


def main():
    """主函数：完整流程演示"""
    print("=" * 60)
    print("🔄 完整流程演示：文档切分 → 向量化 → 存储")
    print("=" * 60)
    
    # 步骤 1：准备文档
    print("\n【步骤 1】准备文档")
    documents = create_sample_documents()
    print(f"✅ 准备了 {len(documents)} 个文档")
    
    # 步骤 2：切分文档
    print("\n【步骤 2】切分文档")
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc["content"], doc["id"], doc["metadata"])
        all_chunks.extend(chunks)
        print(f"  {doc['id']}: {len(chunks)} 个 Chunk")
    print(f"✅ 总共生成 {len(all_chunks)} 个 Chunk")
    
    # 步骤 3：初始化向量数据库
    print("\n【步骤 3】初始化向量数据库")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="chunk_pipeline_demo",
        metadata={"description": "Chunk 切分演示"}
    )
    print(f"✅ 集合已创建: {collection.name}")
    
    # 步骤 4：存储 Chunk
    print("\n【步骤 4】存储 Chunk 到向量数据库")
    store_chunks_to_chroma(all_chunks, collection)
    print(f"✅ 已存储 {collection.count()} 个 Chunk")
    
    # 步骤 5：测试检索
    print("\n【步骤 5】测试检索")
    print("=" * 60)
    
    queries = [
        "Python 有哪些数据类型？",
        "RAG 是什么？",
        "向量数据库有哪些？"
    ]
    
    for query in queries:
        results = search_chunks(collection, query, n_results=2)
        display_search_results(query, results)
    
    # 步骤 6：显示统计信息
    print("\n" + "=" * 60)
    print("📊 统计信息")
    print("=" * 60)
    print(f"总 Chunk 数量: {collection.count()}")
    
    # 按文档统计
    doc_stats = {}
    for chunk in all_chunks:
        source = chunk["metadata"]["source_doc"]
        doc_stats[source] = doc_stats.get(source, 0) + 1
    
    print("\n按文档统计:")
    for doc_id, count in doc_stats.items():
        print(f"  {doc_id}: {count} 个 Chunk")
    
    # 步骤 7：清理说明
    print("\n" + "=" * 60)
    print("✅ 完整流程演示完成！")
    print("=" * 60)
    print("\n💡 关键要点：")
    print("   1. 文档切分是 RAG 的关键步骤")
    print("   2. 切分大小要根据文档类型和检索需求调整")
    print("   3. 元数据（metadata）可以帮助溯源和过滤")
    print("   4. Chroma 会自动处理 Embedding 和索引")


if __name__ == "__main__":
    main()
