# ③ 向量数据库与相似度搜索

## 🧠 概念解释

### 为什么需要向量数据库？

在上一章，我们学会了把文本转成向量。但问题来了：

> **如果有 100 万篇文档，怎么快速找到和用户问题最相似的那几篇？**

朴素方法：遍历所有文档，计算相似度，排序 → **太慢了！**

向量数据库解决的问题：
1. **高效存储**：存储海量向量（百万、千万级别）
2. **快速检索**：在毫秒级找到最相似的向量
3. **索引优化**：用 ANN（近似最近邻）算法加速搜索

### 向量数据库 vs 传统数据库

| 特性 | 传统数据库（MySQL） | 向量数据库（Chroma/FAISS） |
|------|---------------------|---------------------------|
| 存储内容 | 表格数据（数字、字符串） | 向量（高维浮点数组） |
| 查询方式 | 精确匹配（WHERE name = '张三'） | 相似度搜索（找最相近的向量） |
| 查询速度 | B+ 树索引，毫秒级 | ANN 索引，毫秒级 |
| 适用场景 | 业务数据管理 | RAG、推荐系统、图像检索 |

### 主流向量数据库对比

| 数据库 | 特点 | 适用场景 |
|--------|------|----------|
| **FAISS** | Meta 开源，性能强，仅支持本地 | 本地开发、小规模应用 |
| **Chroma** | Python 原生，易上手，支持持久化 | 快速原型、教学、中小项目 |
| **Milvus** | 分布式，企业级，支持海量数据 | 生产环境、大规模应用 |
| **Pinecone** | 云托管，免运维，按量付费 | 快速上线、无运维需求 |
| **Weaviate** | 支持混合检索（向量 + 关键词） | 需要多模态检索的场景 |

**本教程选择 Chroma** — 理由：安装简单、Python 原生、免费开源，适合学习。

### 相似度搜索的核心原理

向量数据库通过 **索引算法** 加速搜索：

```
暴力搜索：计算查询向量与所有向量的距离 → O(n)
ANN 搜索：用数据结构（如 HNSW）快速找到近似最近邻 → O(log n)
```

常见的 ANN 算法：
- **HNSW**（Hierarchical Navigable Small World）：速度快，精度高
- **IVF**（Inverted File Index）：适合海量数据，可调节精度
- **LSH**（Locality-Sensitive Hashing）：用哈希分组，速度快但精度略低

---

## 📦 示例代码

### 示例 1：最简向量数据库（Chroma）

```python
# vector_db_basic.py
# 最简单的向量数据库示例 - 使用 Chroma 最新 API

# 安装依赖：pip install chromadb

import chromadb

# 1. 创建向量数据库（持久化模式）
client = chromadb.PersistentClient(path="./chroma_db")

# 2. 创建集合（相当于"表"）
collection = client.get_or_create_collection(
    name="documents",
    metadata={"description": "我的第一个向量数据库"}
)

# 3. 准备文档
documents = [
    "RAG 是检索增强生成技术，让大模型能基于外部知识回答问题。",
    "向量数据库用于存储和检索文本的向量表示，是 RAG 的核心组件。",
    "Python 是一门流行的编程语言，广泛用于数据科学和 AI 开发。",
    "深度学习是机器学习的一个分支，使用神经网络进行特征学习。",
    "FastAPI 是一个现代的 Python Web 框架，用于构建 API。"
]

# 4. 添加文档（Chroma 会自动使用默认 Embedding 模型）
collection.upsert(
    ids=[f"doc_{i}" for i in range(len(documents))],  # 每个文档需要唯一 ID
    documents=documents
)

print("✅ 文档已添加到向量数据库")
print(f"   当前文档数量：{collection.count()}")
```

**运行方式：**
```bash
cd rag/03_向量数据库与相似度搜索
python vector_db_basic.py
```

---

### 示例 2：相似度搜索实战

```python
# vector_search_demo.py
# 向量相似度搜索实战

import chromadb

# 初始化向量数据库（持久化模式）
client = chromadb.PersistentClient(path="./chroma_db")

# 创建集合
collection = client.get_or_create_collection(name="documents")

# 准备知识库文档
documents = [
    "RAG（检索增强生成）是一种让大模型通过检索外部知识来增强回答的技术。",
    "向量数据库是存储和检索向量的专用数据库，支持相似度搜索。",
    "Embedding 是将文本转换为向量的技术，是 RAG 的基础。",
    "Chunk 是将长文档切分成小块的策略，影响检索质量。",
    "Rerank 是对检索结果进行二次排序的技术，提高相关性。",
    "Prompt 设计是将检索结果喂给大模型的技巧。",
    "Python 是一门流行的编程语言，广泛用于 AI 开发。",
    "FastAPI 是一个高性能的 Python Web 框架。"
]

# 添加文档
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print("=" * 60)
print("📚 知识库构建完成！")
print(f"   文档数量：{collection.count()}")
print("=" * 60)

# 执行检索
queries = [
    "什么是 RAG？",
    "如何让大模型更聪明？",
    "Python 相关技术有哪些？"
]

for query in queries:
    print(f"\n🔍 查询：{query}")
    print("-" * 60)
    
    # 执行相似度搜索
    results = collection.query(
        query_texts=[query],
        n_results=3  # 返回最相似的 3 个文档
    )
    
    # 打印结果
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        similarity = 1 - distance  # Chroma 使用距离，转换为相似度
        print(f"  [{i+1}] 相似度: {similarity:.4f}")
        print(f"      {doc}")
```

**运行方式：**
```bash
cd rag/03_向量数据库与相似度搜索
python vector_search_demo.py
```

**预期输出：**
```
============================================================
📚 知识库构建完成！
   文档数量：8
============================================================

🔍 查询：什么是 RAG？
------------------------------------------------------------
  [1] 相似度: 0.8923
      RAG（检索增强生成）是一种让大模型通过检索外部知识来增强回答的技术。
  [2] 相似度: 0.7645
      向量数据库是存储和检索向量的专用数据库，支持相似度搜索。
  [3] 相似度: 0.7234
      Embedding 是将文本转换为向量的技术，是 RAG 的基础。
```

---

### 示例 3：使用 FAISS 进行高效检索（可选进阶）

```python
# faiss_demo.py
# 使用 FAISS 进行高效向量检索

# 安装依赖：pip install faiss-cpu sentence-transformers

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# 1. 加载 Embedding 模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. 准备文档
documents = [
    "RAG 是检索增强生成技术。",
    "向量数据库用于存储向量。",
    "Python 是一门编程语言。",
    "深度学习使用神经网络。",
    "FastAPI 是 Python Web 框架。"
]

# 3. 生成向量
embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype('float32')

# 4. 构建 FAISS 索引
dimension = embeddings.shape[1]  # 向量维度
index = faiss.IndexFlatL2(dimension)  # 使用 L2 距离
index.add(embeddings)  # 添加向量

print(f"✅ FAISS 索引构建完成，包含 {index.ntotal} 个向量")

# 5. 检索
query = "什么是 RAG？"
query_vector = model.encode([query]).astype('float32')

k = 3  # 返回 top-k
distances, indices = index.search(query_vector, k)

print(f"\n🔍 查询：{query}")
for i, idx in enumerate(indices[0]):
    print(f"  [{i+1}] {documents[idx]} (距离: {distances[0][i]:.4f})")
```

---

## ⚠️ 常见坑

1. **使用过时的 API** — Chroma 最新版本已简化，不再需要 `Settings` 和 `chroma_db_impl` 配置，直接使用 `chromadb.Client()` 或 `chromadb.PersistentClient(path="...")` 即可
2. **忘记持久化** — 使用 `chromadb.Client()` 创建的是内存客户端，重启后数据丢失。生产环境建议使用 `chromadb.PersistentClient(path="...")`
3. **ID 重复** — 每个文档必须有唯一 ID，使用 `upsert()` 可避免重复插入报错
4. **向量维度不一致** — 如果手动指定 Embedding 模型，确保查询和存储使用同一个模型
5. **忽略距离 vs 相似度** — Chroma 返回的是"距离"（越小越相似），可转换为相似度：`相似度 = 1 - 距离`
6. **盲目追求精度** — ANN 算法是"近似"最近邻，不保证完全准确，但速度提升巨大

---

## 🚀 实战建议

1. **开发阶段用 Chroma** — 简单易用，Python 原生
2. **生产环境考虑 Milvus** — 支持分布式、高可用、海量数据
3. **本地性能测试用 FAISS** — 性能最强，但需要手动管理索引
4. **注意文档 ID 设计** — 建议使用业务 ID（如 `article_123`），方便后续更新和删除
5. **监控检索质量** — 定期检查召回率和相关性，必要时调整参数

---

## 📝 小练习

**代码练习：**

1. 运行 `vector_search_demo.py`，尝试修改查询语句，观察检索结果的变化

2. 扩展知识库，添加 10 条新文档，然后测试检索效果

3. **思考题**：如果你的知识库有 100 万条文档，Chroma 还能快速检索吗？需要做什么优化？

4. **进阶练习**：使用 FAISS 示例，尝试不同的索引类型（如 `IndexIVFFlat`），比较搜索速度和精度

---

## 📚 本章知识点汇总

| 知识点名称 | 知识点类型 | 知识点介绍 | 参考链接 |
|-----------|-----------|-----------|---------|
| 向量数据库 | 技术 | 专门用于存储和检索向量的数据库，支持相似度搜索和 ANN 索引 | [Chroma 官网](https://www.trychroma.com/) |
| Chroma | 工具 | 开源的向量数据库，Python 原生，易于上手，适合教学和中小项目 | [Chroma 文档](https://docs.trychroma.com/) |
| FAISS | 工具 | Meta 开源的高效向量检索库，性能强，适合本地开发和大规模检索 | [FAISS GitHub](https://github.com/facebookresearch/faiss) |
| Milvus | 工具 | 企业级分布式向量数据库，支持海量数据和高可用，适合生产环境 | [Milvus 官网](https://milvus.io/) |
| ANN（近似最近邻） | 技术 | 通过牺牲少量精度换取检索速度的算法，如 HNSW、IVF、LSH | [ANN 算法介绍](https://arxiv.org/abs/1603.09320) |
| HNSW | 技术 | 分层导航小世界算法，高效的 ANN 索引方法，速度快精度高 | [HNSW 论文](https://arxiv.org/abs/1603.09320) |
| 余弦距离 | 概念 | 衡量两个向量差异的指标，值越小表示越相似（与余弦相似度相反） | - |
| 持久化 | 概念 | 将内存中的向量数据保存到磁盘，防止重启后丢失 | - |
| Collection | 概念 | Chroma 中的集合概念，类似于关系数据库中的"表" | - |

---

## 📁 相关文件

| 文件名 | 说明 |
|--------|------|
| `README.md` | 本章学习笔记 |
| `vector_db_basic.py` | 最简向量数据库示例（Chroma） |
| `vector_search_demo.py` | 向量相似度搜索实战示例 |
| `faiss_demo.py` | FAISS 高效检索示例（可选） |
