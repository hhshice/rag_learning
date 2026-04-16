# 练习 txt文档 chunk+embedding+存储+相似度搜索
# 使用 chromadb 存储向量数据
# 使用langchain 切分文档

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter


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

def chunk(text,id,metadata = None):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,   # 每个chunk的大小
        chunk_overlap=40,  # chunk之间的重叠大小
        separators=["\n\n", "\n", " ", ""], # 分隔符
        length_function=len  # 计算chunk长度的函数
    )

    chunks = splitter.split_text(text)
    # 构建chunk数据,加上id和metadata
    chunks_data = []
    for i, chunk_content in enumerate(chunks):
        chunk_data = {
            "id": f"{id}_chunk_{i}",
            "content": chunk_content,
            "metadata": {
                **(metadata or {}),
                "source_doc": id,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
        }
        chunks_data.append(chunk_data)
    return chunks_data

def store_chunks_to_chroma(chunks_data,collection):
    ids = [chunk["id"] for chunk in chunks_data ]
    contents = [chunk["content"] for chunk in chunks_data]
    metadatas = [chunk["metadata"] for chunk in chunks_data]

    collection.upsert(
        ids = ids,
        documents = contents,
        metadatas = metadatas
    )




def search_similar_chunks(query, collection, top_k=5):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results


def display_search_results(results):
    for i, (doc_id, content, metadata) in enumerate(zip(results["ids"][0], results["documents"][0], results["metadatas"][0])):
        print(f"结果 {i+1}:")
        print(f"ID: {doc_id}")
        print(f"内容: {content}")
        print(f"元数据: {metadata}")
        print("-" * 50)

if __name__ == "__main__":
    # 初始化向量数据库
    client = chromadb.PersistentClient(path="./chroma_db")
    # 创建集合
    # 如果集合不存在，则创建一个新的集合，否则返回已存在的集合
    collection = client.get_or_create_collection(
        name="my_collection"
        # 这里没有指定embedding_function，使用默认的embedding函数，默认使用OpenAI的embedding模型
        # 但是这样不好，需要优化
    )
    # 创建示例文档
    documents = create_sample_documents()
    # 处理每个文档
    for doc in documents:
        doc_id = doc["id"]
        content = doc["content"]
        metadata = doc["metadata"]
        # 切分文档
        chunks_data = chunk(content, doc_id, metadata)
        # 存储到向量数据库
        store_chunks_to_chroma(chunks_data, collection)
    # 测试相似度搜索
    query = "什么是RAG？"
    results = search_similar_chunks(query, collection)
    display_search_results(results)


    






