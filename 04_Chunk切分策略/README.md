# ④ Chunk 切分策略（重点）

## 🧠 概念解释

### 什么是 Chunk（文档切分）？

> **Chunk = 把长文档切成小块，让向量数据库能够精确检索**

举个例子：

```
原始文档（5000字）：
"公司成立于2010年...（大量内容）...我们的退换货政策如下：7天内无理由退货..."

问题：
用户问："退换货政策是什么？"
❌ 如果不切分：整个文档作为一个检索单元 → 可能检索不到或检索到大量无关内容
✅ 切分后：每个 Chunk 是独立的小段落 → 精准匹配到"退换货政策"相关段落
```

### 为什么需要切分？

| 问题 | 不切分的后果 |
|------|-------------|
| **文档太长** | Embedding 模型有 token 限制（如 8192 tokens），超长文本会被截断或效果差 |
| **检索精度低** | 长文档包含多个主题，查询时难以精准匹配到具体段落 |
| **上下文噪音** | 返回的文档包含大量无关信息，干扰大模型理解 |
| **成本问题** | 文档越长，Embedding 计算成本越高，存储空间越大 |

**核心原则：**
> **Chunk 大小要平衡两个矛盾需求：**
> - 太小：丢失上下文，语义不完整
> - 太大：检索精度下降，包含过多噪音

### 常见的切分策略

| 策略 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| **固定长度切分** | 按字符数或 token 数切分 | 简单快速 | 可能切断句子，语义不完整 | 结构化文本、代码 |
| **句子级切分** | 按标点符号（句号、问号）切分 | 语义完整 | Chunk 可能过短 | 短文档、问答对 |
| **段落级切分** | 按换行符或段落标记切分 | 语义完整，上下文好 | 段落长度不均 | 文章、报告 |
| **递归字符切分** | 按优先级依次尝试不同分隔符 | 灵活，语义较好 | 需要调参 | 通用场景 |
| **语义切分** | 基于 Embedding 相似度切分 | 语义最佳 | 计算成本高 | 高精度需求 |
| **滑动窗口** | 固定窗口大小 + 重叠部分 | 保留上下文 | 存储冗余 | 需要上下文的场景 |

### 切分参数详解

**核心参数：**

```python
chunk_size = 500      # 每个 Chunk 的最大长度（字符数或 token 数）
chunk_overlap = 50    # 相邻 Chunk 之间的重叠部分
```

**为什么要重叠？**

```
文档：A...B...C...D...

不重叠：
  Chunk1: A...B
  Chunk2: C...D
  ❌ B 和 C 的语义被切断

重叠：
  Chunk1: A...B...C
  Chunk2: B...C...D
  ✅ B 和 C 在两个 Chunk 中都出现，保留完整上下文
```

### 不同的切分工具

| 工具 | 特点 | 推荐度 |
|------|------|--------|
| **LangChain** | 提供多种切分器，功能全面 | ⭐⭐⭐⭐⭐ |
| **LlamaIndex** | 语义切分效果好 | ⭐⭐⭐⭐ |
| **自定义切分** | 完全可控，适合特殊需求 | ⭐⭐⭐ |

---

## 📦 示例代码

### 示例 1：固定长度切分（最简单）

```python
# chunk_fixed_size.py
# 固定长度切分示例

def fixed_size_chunk(text, chunk_size=100, overlap=20):
    """
    按固定长度切分文本
    
    Args:
        text: 原始文本
        chunk_size: 每个 Chunk 的大小（字符数）
        overlap: 相邻 Chunk 的重叠字符数
    
    Returns:
        chunks: 切分后的 Chunk 列表
    """
    chunks = []
    start = 0
    
    while start < len(text):
        # 计算当前 Chunk 的结束位置
        end = start + chunk_size
        
        # 提取 Chunk
        chunk = text[start:end]
        chunks.append(chunk)
        
        # 移动到下一个 Chunk 的起始位置（考虑重叠）
        start = end - overlap
        
        # 如果剩余文本不足一个 Chunk，直接结束
        if end >= len(text):
            break
    
    return chunks


# 测试
text = """
RAG（检索增强生成）是一种让大模型通过检索外部知识来增强回答的技术。
它的核心流程包括：文档加载、文档切分、向量化、存储、检索、生成。
其中，文档切分是一个关键步骤，直接影响检索质量。
好的切分策略应该平衡语义完整性和检索精度。
"""

chunks = fixed_size_chunk(text, chunk_size=50, overlap=10)

print("=" * 60)
print("📄 固定长度切分结果")
print("=" * 60)
for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i} (长度: {len(chunk)}):")
    print(f"  {repr(chunk)}")
```

---

### 示例 2：使用 LangChain 的递归字符切分器（推荐）

```python
# chunk_langchain_demo.py
# 使用 LangChain 进行文档切分（推荐方法）

# 安装依赖：pip install langchain-text-splitters

from langchain_text_splitters import RecursiveCharacterTextSplitter

def demo_recursive_chunk():
    """
    演示 LangChain 的递归字符切分器
    """
    # 准备示例文本
    text = """
# 公司退换货政策

## 第一章 总则

本公司所有商品均支持7天无理由退换货。退换货政策适用于在官网购买的所有商品。

## 第二章 退货流程

### 2.1 申请退货

用户可在订单详情页面点击"申请退货"按钮，填写退货原因并提交申请。我们将在1-3个工作日内处理您的申请。

### 2.2 寄回商品

退货申请通过后，请在7天内将商品寄回。建议使用快递并保留快递单号，以便查询物流信息。

## 第三章 退款说明

退款将在确认收到商品后的3-5个工作日内原路返回。如果是质量问题，运费由我们承担；如果是个人原因，运费由用户承担。
"""
    
    # 创建切分器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,          # 每个 Chunk 的最大字符数
        chunk_overlap=20,        # 相邻 Chunk 的重叠字符数
        length_function=len,     # 计算长度的函数
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]  # 分隔符优先级
    )
    
    # 执行切分
    chunks = splitter.split_text(text)
    
    print("=" * 60)
    print("📄 递归字符切分结果（LangChain）")
    print("=" * 60)
    print(f"原始文本长度: {len(text)} 字符")
    print(f"切分后 Chunk 数量: {len(chunks)}")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} (长度: {len(chunk)}) ---")
        print(chunk.strip())


if __name__ == "__main__":
    demo_recursive_chunk()
```

---

### 示例 3：语义切分（进阶）

```python
# chunk_semantic.py
# 语义切分示例 - 基于 Embedding 相似度

# 安装依赖：pip install sentence-transformers numpy

from sentence_transformers import SentenceTransformer
import numpy as np

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def semantic_chunk(text, model_name='all-MiniLM-L6-v2', threshold=0.7):
    """
    基于语义相似度的文档切分
    
    原理：
    1. 将文本按句子切分
    2. 计算相邻句子的相似度
    3. 相似度低的句子之间作为切分点
    
    Args:
        text: 原始文本
        model_name: Embedding 模型名称
        threshold: 相似度阈值，低于此值则切分
    
    Returns:
        chunks: 切分后的 Chunk 列表
    """
    # 加载模型
    model = SentenceTransformer(model_name)
    
    # 按句子切分（简单实现，实际可用更复杂的分句逻辑）
    sentences = [s.strip() for s in text.split('。') if s.strip()]
    
    if len(sentences) == 0:
        return [text]
    
    # 生成句子向量
    embeddings = model.encode(sentences)
    
    # 计算相邻句子的相似度
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # 计算当前句子与前一个句子的相似度
        sim = cosine_similarity(embeddings[i-1], embeddings[i])
        
        if sim < threshold:
            # 相似度低，在此切分
            chunks.append('。'.join(current_chunk) + '。')
            current_chunk = [sentences[i]]
        else:
            # 相似度高，继续合并
            current_chunk.append(sentences[i])
    
    # 添加最后一个 Chunk
    if current_chunk:
        chunks.append('。'.join(current_chunk) + '。')
    
    return chunks


def demo_semantic_chunk():
    """演示语义切分"""
    text = """
机器学习是人工智能的一个分支。它使用统计技术让计算机系统能够从数据中学习。
深度学习是机器学习的子领域。它使用多层神经网络来学习数据的层次化表示。
自然语言处理是AI的重要应用领域。它让计算机能够理解和生成人类语言。
计算机视觉让机器能够看懂图像和视频。它在自动驾驶、医疗影像等领域有广泛应用。
"""
    
    print("=" * 60)
    print("🧠 语义切分结果")
    print("=" * 60)
    
    chunks = semantic_chunk(text, threshold=0.5)
    
    print(f"原始文本长度: {len(text)} 字符")
    print(f"切分后 Chunk 数量: {len(chunks)}")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk.strip())


if __name__ == "__main__":
    demo_semantic_chunk()
```

---

### 示例 4：完整流程 - 切分 + 向量化 + 存储

```python
# chunk_complete_pipeline.py
# 完整流程：文档切分 → 向量化 → 存储到向量数据库

# 安装依赖：pip install chromadb langchain-text-splitters

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_and_store(text, collection, doc_id_prefix="doc"):
    """
    完整流程：切分文档并存储到向量数据库
    
    Args:
        text: 原始文本
        collection: Chroma 集合对象
        doc_id_prefix: 文档 ID 前缀
    """
    # 1. 创建切分器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,       # 每个 Chunk 最大 200 字符
        chunk_overlap=40,     # 重叠 40 字符
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    
    # 2. 执行切分
    chunks = splitter.split_text(text)
    
    print(f"📄 文档切分完成：{len(chunks)} 个 Chunk")
    
    # 3. 存储到向量数据库
    chunk_ids = [f"{doc_id_prefix}_chunk_{i}" for i in range(len(chunks))]
    
    collection.upsert(
        ids=chunk_ids,
        documents=chunks,
        metadatas=[{"source": doc_id_prefix, "chunk_index": i} for i in range(len(chunks))]
    )
    
    print(f"✅ 已存储到向量数据库")
    return chunks


def demo_complete_pipeline():
    """演示完整流程"""
    # 准备文档
    documents = [
        """
# Python 编程入门

Python 是一门流行的编程语言。它的语法简洁，易于学习。
Python 广泛应用于 Web 开发、数据科学、人工智能等领域。

## 变量与数据类型

Python 支持多种数据类型：整数、浮点数、字符串、列表、字典等。
变量无需声明类型，Python 会自动推断。
        """,
        """
# RAG 技术详解

RAG 是检索增强生成的缩写。它让大模型能够通过检索外部知识来增强回答。
RAG 的核心组件包括：文档切分、向量数据库、Embedding 模型、检索算法。

## 应用场景

RAG 适用于企业知识库、智能客服、文档问答等场景。
        """
    ]
    
    # 初始化向量数据库
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="chunk_demo")
    
    print("=" * 60)
    print("🔄 完整流程演示：文档切分 → 向量化 → 存储")
    print("=" * 60)
    
    # 处理每个文档
    all_chunks = []
    for i, doc in enumerate(documents):
        print(f"\n处理文档 {i+1}...")
        chunks = chunk_and_store(doc, collection, doc_id_prefix=f"doc_{i}")
        all_chunks.extend(chunks)
    
    # 演示检索
    print("\n" + "=" * 60)
    print("🔍 测试检索")
    print("=" * 60)
    
    query = "Python 有哪些数据类型？"
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    print(f"查询：{query}")
    print("\n最相关的 Chunk：")
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0]), 1):
        similarity = 1 - distance
        print(f"\n[{i}] 相似度: {similarity:.4f}")
        print(f"内容: {doc[:100]}...")
    
    print("\n" + "=" * 60)
    print("✅ 演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    demo_complete_pipeline()
```

---

## ⚠️ 常见坑

1. **Chunk 太小（< 100 字符）** — 丢失上下文，语义不完整，检索效果差
2. **Chunk 太大（> 1000 字符）** — 检索精度下降，包含过多无关信息
3. **忽略重叠** — 不使用 `chunk_overlap` 会导致语义被切断
4. **分隔符选择不当** — 中文文本应该用中文标点作为分隔符，不能用英文标点
5. **盲目使用固定长度切分** — 对于结构化文档（如 Markdown），应该利用标题等结构信息
6. **不考虑 Embedding 模型的限制** — 不同模型的 token 限制不同，切分时要考虑
7. **忽略元数据** — 每个 Chunk 应该附带元数据（如来源文档、页码），方便溯源

---

## 🚀 实战建议

### 1. 如何选择 Chunk 大小？

| 文档类型 | 推荐 chunk_size | 推荐 overlap |
|---------|----------------|--------------|
| 新闻文章 | 300-500 | 50-100 |
| 技术文档 | 500-1000 | 100-200 |
| 问答对 | 100-300 | 0-50 |
| 代码文件 | 200-500 | 50-100 |
| 法律/医疗文档 | 1000-2000 | 200-400 |

### 2. 切分策略选择指南

```
文档类型 → 切分策略

结构化文档（Markdown、HTML）
  → 优先按标题/段落切分
  
非结构化文档（纯文本）
  → 递归字符切分
  
代码文件
  → 按函数/类切分
  
需要高精度
  → 语义切分
```

### 3. 生产环境检查清单

- [ ] 检查文档的最大 token 数是否超过 Embedding 模型限制
- [ ] 为每个 Chunk 添加元数据（来源、页码、章节）
- [ ] 测试切分效果：随机抽样检查 Chunk 语义是否完整
- [ ] 根据检索效果调整 `chunk_size` 和 `chunk_overlap`
- [ ] 记录切分参数，方便复现

---

## 📝 小练习

**代码练习：**

1. 运行 `chunk_langchain_demo.py`，观察切分结果，尝试调整 `chunk_size` 和 `chunk_overlap` 参数

2. 准备一篇长文章（至少 2000 字），分别用固定长度切分和递归字符切分，比较结果差异

3. **思考题**：
   - 如果文档包含表格和图片，应该如何切分？
   - Chunk 的 overlap 设置为多少比较合适？太大或太小有什么问题？
   - 如何评估切分效果的好坏？

4. **进阶练习**：实现一个切分器，能够识别 Markdown 标题并按章节切分

---

## 📚 本章知识点汇总

| 知识点名称 | 知识点类型 | 知识点介绍 | 参考链接 |
|-----------|-----------|-----------|---------|
| Chunk（文档切分） | 概念 | 将长文档切分成小块的过程，是 RAG 流程的关键步骤 | - |
| chunk_size | 技术参数 | 每个 Chunk 的最大长度（字符数或 token 数） | - |
| chunk_overlap | 技术参数 | 相邻 Chunk 之间的重叠部分，保留上下文 | - |
| 固定长度切分 | 技术 | 按固定字符数或 token 数切分文档 | - |
| 递归字符切分 | 技术 | 按优先级依次尝试不同分隔符，灵活且语义较好 | [LangChain 文档](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter) |
| 语义切分 | 技术 | 基于 Embedding 相似度切分文档，语义最佳但计算成本高 | - |
| LangChain | 框架 | 提供丰富的文档切分工具，是 RAG 开发的主流框架 | [LangChain 官网](https://www.langchain.com/) |
| RecursiveCharacterTextSplitter | 工具类 | LangChain 提供的递归字符切分器，推荐用于通用场景 | [API 文档](https://api.python.langchain.com/en/latest/text_splitters/langchain_text_splitters.base.RecursiveCharacterTextSplitter.html) |
| Token 限制 | 概念 | Embedding 模型对输入长度的限制，通常为 512-8192 tokens | - |
| 元数据（Metadata） | 概念 | 每个 Chunk 附带的信息，如来源文档、页码、章节等 | - |

---

## 📁 相关文件

| 文件名 | 说明 |
|--------|------|
| `README.md` | 本章学习笔记 |
| `chunk_fixed_size.py` | 固定长度切分示例 |
| `chunk_langchain_demo.py` | LangChain 递归字符切分示例（推荐） |
| `chunk_semantic.py` | 语义切分示例（进阶） |
| `chunk_complete_pipeline.py` | 完整流程示例：切分 → 向量化 → 存储 |
