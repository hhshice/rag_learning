# ⑧ 搭建一个最小 RAG Demo

## 🧠 概念解释

### 这个 Demo 包含什么？

我们将搭建一个**完整可运行的 RAG 系统**，整合前面学到的所有知识点：

```
┌─────────────────────────────────────────────────────────────┐
│                      完整 RAG 系统架构                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  【知识库构建阶段】                                           │
│    原始文档 → 文档切分 → Embedding → 向量数据库               │
│                                                             │
│  【检索阶段】                                                 │
│    用户问题 → Query预处理 → 向量检索 → Rerank                │
│                                                             │
│  【生成阶段】                                                 │
│    检索结果 + 问题 → Prompt构建 → 大模型 → 最终回答           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Demo 功能特性

| 功能模块 | 技术实现 | 说明 |
|---------|---------|------|
| 文档处理 | LangChain TextSplitter | 支持多种文档格式 |
| 向量存储 | Chroma | 持久化存储，快速检索 |
| Embedding | sentence-transformers | 本地免费，中文效果好 |
| 检索 | 向量检索 + 混合检索 | 支持多种检索策略 |
| Rerank | Cross-Encoder | 提升检索精度 |
| Prompt | 动态模板 | 多种风格可选 |
| 接口 | 命令行 / Web API | 灵活使用方式 |

### 项目结构

```
08_搭建最小RAG_Demo/
├── README.md                    # 本文档
├── mini_rag.py                  # 核心 RAG 类
├── demo_cli.py                  # 命令行演示
├── demo_web.py                  # Web API (FastAPI)
├── knowledge_base/              # 知识库目录
│   └── sample_docs.txt          # 示例文档
└── requirements.txt             # 依赖列表
```

---

## 📦 核心代码实现

### 核心类：MiniRAG

```python
# mini_rag.py
# 最小 RAG 系统核心实现

# 安装依赖：
# pip install chromadb sentence-transformers langchain-text-splitters

import chromadb
from chromadb.config import Settings
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
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        初始化 RAG 系统
        
        Args:
            persist_dir: 向量数据库持久化目录
            collection_name: 集合名称
            embedding_model: Embedding 模型名称
            rerank_model: Rerank 模型名称
        """
        # 初始化向量数据库
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG Demo 知识库"}
        )
        
        # 加载 Embedding 模型
        print(f"🔄 加载 Embedding 模型: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # 加载 Rerank 模型
        print(f"🔄 加载 Rerank 模型: {rerank_model}")
        self.rerank_model = CrossEncoder(rerank_model)
        
        # 文档切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        
        print("✅ RAG 系统初始化完成")
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None
    ):
        """
        添加文档到知识库
        
        Args:
            documents: 文档列表
            metadatas: 元数据列表（可选）
        """
        print(f"📄 添加 {len(documents)} 个文档...")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for i, doc in enumerate(documents):
            # 切分文档
            chunks = self.text_splitter.split_text(doc)
            
            for j, chunk in enumerate(chunks):
                chunk_id = f"doc_{i}_chunk_{j}"
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                
                # 元数据
                if metadatas and i < len(metadatas):
                    meta = {**metadatas[i], "chunk_index": j}
                else:
                    meta = {"doc_index": i, "chunk_index": j}
                all_metadatas.append(meta)
        
        # 添加到向量数据库
        self.collection.upsert(
            ids=all_ids,
            documents=all_chunks,
            metadatas=all_metadatas
        )
        
        print(f"✅ 已添加 {len(all_chunks)} 个 Chunk")
        print(f"   知识库总文档数：{self.collection.count()}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回文档数量
            use_rerank: 是否使用 Rerank
        
        Returns:
            results: 检索结果列表
        """
        # 向量检索
        retrieve_k = top_k * 3 if use_rerank else top_k
        
        search_results = self.collection.query(
            query_texts=[query],
            n_results=retrieve_k
        )
        
        documents = search_results['documents'][0]
        ids = search_results['ids'][0]
        distances = search_results['distances'][0]
        metadatas = search_results['metadatas'][0]
        
        # Rerank
        if use_rerank:
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
            style: 风格（strict/balanced/creative）
        
        Returns:
            prompt: 构建好的 Prompt
        """
        # 格式化文档
        docs_text = "\n\n".join([
            f"【参考资料{i+1}】\n{doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        if style == "strict":
            return f"""你是一个专业的问答助手。

【重要规则】
1. 只使用参考资料中的信息回答
2. 如果参考资料中没有相关信息，请明确说明
3. 不要编造或推测任何内容
4. 回答要简洁准确

参考资料：
{docs_text}

用户问题：{query}

请根据规则回答："""
        
        elif style == "balanced":
            return f"""你是一个专业的问答助手。

请根据参考资料回答用户问题。如果参考资料信息不足，可以结合常识补充，但要明确标注。

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
        prompt_style: str = "balanced"
    ) -> Dict[str, Any]:
        """
        完整查询流程
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
            use_rerank: 是否使用 Rerank
            prompt_style: Prompt 风格
        
        Returns:
            result: 包含 prompt 和检索结果的字典
        """
        # 1. 检索
        print(f"\n🔍 检索：{question}")
        documents = self.retrieve(question, top_k=top_k, use_rerank=use_rerank)
        
        # 2. 构建 Prompt
        prompt = self.build_prompt(question, documents, style=prompt_style)
        
        return {
            "question": question,
            "documents": documents,
            "prompt": prompt
        }
```

---

### 命令行演示

```python
# demo_cli.py
# 命令行 RAG 演示

from mini_rag import MiniRAG


def create_sample_knowledge_base():
    """创建示例知识库"""
    
    documents = [
        """
# RAG 技术介绍

RAG（检索增强生成）是一种让大模型通过检索外部知识来增强回答的技术。
它的核心思想是让大模型"开卷考试"，先从知识库中找到相关内容，再基于找到的内容生成回答。

RAG 的主要优势包括：
1. 避免幻觉：大模型基于真实文档回答，减少编造
2. 知识更新：无需重新训练模型，只需更新知识库
3. 可溯源：可以追踪回答的来源文档

RAG 适用于企业知识库、智能客服、文档问答等场景。
        """,
        """
# 向量数据库

向量数据库是专门用于存储和检索向量的数据库。它支持高效的相似度搜索，是 RAG 系统的核心组件。

常见的向量数据库包括：
- Chroma：轻量级，易于上手
- FAISS：Facebook 开源，性能强大
- Milvus：企业级，支持分布式
- Pinecone：云服务，免运维

向量数据库的核心功能是将文本转换为向量，并支持快速检索最相似的向量。
        """,
        """
# Embedding 模型

Embedding 是将文本转换为向量的技术。通过 Embedding，文本的语义信息可以用数值表示，从而支持相似度计算。

常用的 Embedding 模型：
- all-MiniLM-L6-v2：速度快，效果好
- text-embedding-ada-002：OpenAI 官方，效果好
- m3e-base：中文效果好

Embedding 模型的选择会影响检索效果，需要根据具体场景选择。
        """,
        """
# 文档切分策略

文档切分是 RAG 的关键步骤，直接影响检索效果。

切分策略：
- 固定长度切分：简单快速，但可能切断语义
- 递归字符切分：按优先级尝试不同分隔符，语义较好
- 语义切分：基于 Embedding 相似度，语义最佳

推荐参数：
- chunk_size: 200-500 字符
- chunk_overlap: 10-20%
        """,
        """
# Rerank 技术

Rerank 是对检索结果进行二次排序的技术，可以显著提升检索精度。

两阶段检索架构：
1. 召回阶段：向量检索召回 20-50 个文档
2. 精排阶段：Cross-Encoder 对召回文档精确排序

Rerank 的优势：
- 提升相关性：Cross-Encoder 能捕捉深层语义
- 过滤噪音：通过分数阈值过滤低质量结果

推荐使用 Cross-Encoder 模型进行 Rerank。
        """
    ]
    
    metadatas = [
        {"source": "RAG技术介绍", "category": "基础概念"},
        {"source": "向量数据库", "category": "技术组件"},
        {"source": "Embedding模型", "category": "技术组件"},
        {"source": "文档切分策略", "category": "最佳实践"},
        {"source": "Rerank技术", "category": "优化方法"}
    ]
    
    return documents, metadatas


def main():
    """主函数"""
    
    print("=" * 60)
    print("🎯 最小 RAG Demo - 命令行版本")
    print("=" * 60)
    
    # 初始化 RAG 系统
    rag = MiniRAG(
        persist_dir="./demo_chroma_db",
        collection_name="demo_kb"
    )
    
    # 创建知识库
    documents, metadatas = create_sample_knowledge_base()
    rag.add_documents(documents, metadatas)
    
    print("\n" + "=" * 60)
    print("💬 开始问答（输入 'quit' 退出）")
    print("=" * 60)
    
    while True:
        question = input("\n请输入问题：").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 再见！")
            break
        
        if not question:
            continue
        
        # 执行查询
        result = rag.query(
            question,
            top_k=3,
            use_rerank=True,
            prompt_style="balanced"
        )
        
        # 显示检索结果
        print("\n📚 检索结果：")
        print("-" * 60)
        for i, doc in enumerate(result['documents'], 1):
            print(f"  [{i}] 来源：{doc['metadata'].get('source', '未知')}")
            print(f"      相似度：{doc['vector_similarity']:.4f}")
            if 'rerank_score' in doc:
                print(f"      Rerank分数：{doc['rerank_score']:.4f}")
            print(f"      内容：{doc['content'][:80]}...")
        
        # 显示构建的 Prompt
        print("\n📝 构建的 Prompt：")
        print("-" * 60)
        print(result['prompt'])
        
        print("\n💡 提示：将上述 Prompt 发送给大模型（如 GPT-4）即可获得回答")


if __name__ == "__main__":
    main()
```

---

## ⚠️ 常见问题

1. **内存不足** — Embedding 和 Rerank 模型需要较多内存，建议至少 4GB
2. **首次运行慢** — 需要下载模型，首次运行会较慢
3. **检索结果不理想** — 尝试调整 `chunk_size`、`top_k` 参数
4. **中文效果差** — 使用中文 Embedding 模型（如 m3e-base）

---

## 🚀 运行方式

```bash
# 安装依赖
pip install -r requirements.txt

# 运行命令行版本
python demo_cli.py

# 运行 Web API 版本（需要安装 FastAPI）
python demo_web.py
```

---

## 📚 本章知识点汇总

| 知识点名称 | 知识点类型 | 知识点介绍 | 参考链接 |
|-----------|-----------|-----------|---------|
| RAG 系统架构 | 概念 | 完整的检索增强生成系统，包含文档处理、检索、生成三个阶段 | - |
| Chroma | 工具 | 开源向量数据库，Python 原生，易于使用 | [Chroma 官网](https://www.trychroma.com/) |
| sentence-transformers | 框架 | 开源的 Sentence Embedding 库，提供多种预训练模型 | [GitHub](https://github.com/UKPLab/sentence-transformers) |
| LangChain | 框架 | RAG 开发的主流框架，提供丰富的工具和组件 | [LangChain 官网](https://www.langchain.com/) |
| FastAPI | 框架 | 现代高性能 Python Web 框架，适合构建 API | [FastAPI 官网](https://fastapi.tiangolo.com/) |

---

## 📁 相关文件

| 文件名 | 说明 |
|--------|------|
| `README.md` | 本文档 |
| `mini_rag.py` | 核心 RAG 类实现 |
| `demo_cli.py` | 命令行演示 |
| `demo_web.py` | Web API 演示 |
| `requirements.txt` | 依赖列表 |
