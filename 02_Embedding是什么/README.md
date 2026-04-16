# ② Embedding 是什么（向量化本质）

## 🧠 概念解释

### 什么是 Embedding？

**Embedding = 把文字变成一串数字（向量），让计算机能"理解"文字的含义**

回顾上一章的问题：关键词匹配有缺陷——无法理解语义。

例如：
- 用户搜："如何修电脑"
- 文档里有："笔记本维修指南"

关键词匹配：没有重叠词 → 匹配失败 ❌
人类理解：有重叠语义 → 匹配成功 ✅

**Embedding 就是让计算机学会这种"语义理解"的技术。**

### Embedding 的直观理解

把每个词或每句话转换成一个**向量**（一串数字）。

举例（简化版，非真实数据）：

| 文本 | 向量表示（简化示例） |
|------|---------------------|
| "狗" | [0.9, 0.1, 0.2, ...] |
| "猫" | [0.85, 0.15, 0.18, ...] |
| "汽车" | [0.1, 0.9, 0.8, ...] |

可以看到：
- "狗"和"猫"的向量很接近（都是动物）
- "狗"和"汽车"的向量距离远（语义不相关）

###  Embedding 如何判断相似度？

把文本变成向量后，通过计算**向量之间的距离**来判断相似度：

| 方法 | 公式 | 特点 |
|------|------|------|
| **余弦相似度** | cos(θ) | 最常用，衡量方向相似性，范围 [-1, 1] |
| **欧氏距离** | √(Σ(a-b)²) | 几何距离，直观 |
| **点积** | Σ(a×b) | 计算快，与余弦类似 |

**余弦相似度举例：**
```
向量 A = [1, 0]
向量 B = [1, 0]
cos(A, B) = 1.0  → 完全相同

向量 A = [1, 0]
向量 B = [0, 1]
cos(A, B) = 0.0  → 完全垂直

向量 A = [1, 0]
向量 B = [-1, 0]
cos(A, B) = -1.0 → 完全相反
```

### Embedding 在 RAG 中的位置？

```
文档处理阶段：文本 → Embedding模型 → 向量 → 存入向量数据库
查询阶段：用户问题 → Embedding模型 → 查询向量 → 在向量数据库中检索相似文档
```

---

## 📦 示例代码

### 原理演示：字符级相似度（无需任何依赖）

以下代码用最简单的方式演示 Embedding 的核心原理：

```python
# embedding_demo.py
# Embedding 原理演示 - 用字符重叠模拟语义相似度
# 无需 API Key，通过简单算法直观理解 Embedding 的工作原理

def simple_tokenize(text):
    """简单分词：按字符拆分"""
    return set(text.lower())

def cosine_similarity(a, b):
    """计算两个集合的 Jaccard 相似度（简化版）"""
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0

def main():
    documents = [
        "如何修理笔记本电脑",
        "汽车发动机故障排查",
        "台式机内存升级教程",
        "餐厅美食推荐",
        "Python编程入门",
        "机器学习算法原理",
        "电脑屏幕不显示",
        "川菜麻辣菜谱"
    ]

    query = "电脑坏了怎么修"
    query_chars = simple_tokenize(query)

    # 计算相似度
    similarities = []
    for doc in documents:
        doc_chars = simple_tokenize(doc)
        sim = cosine_similarity(query_chars, doc_chars)
        similarities.append((sim, doc))

    similarities.sort(reverse=True)

    for i, (sim, doc) in enumerate(similarities, 1):
        print(f"  [{i}] 相似度: {sim:.4f} - {doc}")

if __name__ == "__main__":
    main()
```

**运行方式：**
```bash
cd rag/02_Embedding是什么
python embedding_demo.py
```

**运行结果：**
```
============================================================
📊 相似度排序结果：

  [1] ██████░░░░░░░░░░░░░░░░░░░░░░░ 0.2308  如何修理笔记本电脑
      公共字符: {'修', '脑', '电'}
  [2] █████░░░░░░░░░░░░░░░░░░░░░░░░ 0.1667  电脑屏幕不显示
      公共字符: {'脑', '电'}
  [3] ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0000  餐厅美食推荐
  ...
============================================================
✅ 关键发现：
   '电脑坏了怎么修' 与 '如何修理笔记本电脑' 最相似
```

### 进阶：使用真实的 Embedding 模型

如果你想使用真实的深度学习 Embedding，有以下选择：

**方案 1：sentence-transformers（推荐，免费开源）**
```bash
pip install sentence-transformers
```
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["你好", "世界"])
```

**方案 2：OpenAI API（需要付费）**
```bash
pip install openai
```
```python
import openai
openai.api_key = "your-api-key"
response = openai.Embedding.create(
    input="你的文本",
    model="text-embedding-ada-002"
)
embedding = response['data'][0]['embedding']
```

**方案 3：中文开源模型（推荐国内使用）**
```python
# 使用 M3E 模型，中文效果好
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('moka-ai/m3e-base')
```

---

## ⚠️ 常见坑

1. **Embedding 模型选错** — 不同模型擅长不同任务，例如 `text-embedding-ada-002` 适合通用场景，代码专用模型要用 `code-search-*`
2. **向量维度不一致** — 确保比较的向量来自同一个模型，否则没有意义
3. **忽略 Embedding 长度限制** — 大多数模型有 token 限制（如 8192 tokens），超长文本需要截断或分段
4. **相似度阈值设置不当** — 相似度阈值太低会召回无关内容，太高会漏掉相关内容。需要在实际数据上调优

---

## 🚀 实战建议

1. **先免费后付费** — 开发阶段用 `sentence-transformers`（免费），生产环境根据需求考虑 OpenAI API 或其他商业方案
2. **选择合适的模型** — 不同 Embedding 模型的效果和速度差异很大，常见选择：
   - `all-MiniLM-L6-v2`：速度快，效果不错（适合入门）
   - `text-embedding-ada-002`：OpenAI 官方，效果好（需要付费）
   - `m3e-base`：国产开源，中文效果好
3. **批量处理更高效** — 如果有大量文档，用批量 API 而不是循环调用

---

## 📝 小练习

**代码练习：**

1. 修改上面的代码，添加一个新的文档："Python 编程入门"，然后查询"怎么写代码"，看看结果是否符合预期

2. 尝试计算两个任意文本之间的相似度，理解余弦相似度的取值范围

3. **思考题**：如果你的知识库包含中英文混合内容，普通的 Embedding 模型能正确处理吗？有什么解决方案？

---

## 📚 本章知识点汇总

| 知识点名称 | 知识点类型 | 知识点介绍 | 参考链接 |
|-----------|-----------|-----------|---------|
| Embedding（嵌入向量） | 技术 | 将文本转换为固定维度向量的技术，使文本的语义信息能用数值表示 | [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) |
| 向量（Vector） | 概念 | 一串有序的数字，可以理解为多维空间中的一个点 | - |
| 余弦相似度 | 技术 | 通过计算两个向量夹角的余弦值来衡量相似度，范围 [-1, 1] | - |
| 欧氏距离 | 技术 | 多维空间中两点之间的直线距离 | - |
| sentence-transformers | 框架 | 开源的 Sentence Embedding 库，基于 Transformer 模型 | [GitHub](https://github.com/UKPLab/sentence-transformers) |
| text-embedding-ada-002 | 工具 | OpenAI 官方 Embedding 模型，通用场景效果好 | [OpenAI 官方文档](https://platform.openai.com/docs/guides/embeddings) |
| Token | 概念 | 文本处理的基本单位，中文约 1-2 个字符 = 1 Token | - |

---

## 📁 相关文件

| 文件名 | 说明 |
|--------|------|
| `README.md` | 本章学习笔记 |
| `embedding_local.py` | 使用 sentence-transformers 的本地示例（推荐，无需 API Key） |
| `embedding_demo.py` | 使用 OpenAI API 的示例（需要 API Key） |
