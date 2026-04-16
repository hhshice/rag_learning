# ⑦ Prompt 设计（如何喂给大模型）

## 🧠 概念解释

### 什么是 Prompt 设计？

> **Prompt 设计 = 将检索结果和用户问题组织成合适的格式，喂给大模型生成回答**

在 RAG 系统中，Prompt 设计是连接检索和生成的桥梁。

```
检索结果 + 用户问题 → [Prompt 设计] → 大模型 → 最终回答
```

### 为什么 Prompt 设计很重要？

| 问题 | 糟糕的 Prompt | 好的 Prompt |
|------|-------------|------------|
| **回答不相关** | 简单拼接检索结果 | 明确指示"根据参考资料回答" |
| **幻觉严重** | 没有限制大模型行为 | 明确要求"不要编造信息" |
| **格式混乱** | 无格式要求 | 指定输出格式（如分点作答） |
| **遗漏关键信息** | 没有引导关注重点 | 明确指出需要回答的要点 |

### Prompt 的核心组成部分

```
┌─────────────────────────────────────────────┐
│                  Prompt 结构                 │
├─────────────────────────────────────────────┤
│                                             │
│  【1. 角色设定】                             │
│   "你是一个专业的问答助手..."                │
│                                             │
│  【2. 任务说明】                             │
│   "请根据以下参考资料回答用户问题..."        │
│                                             │
│  【3. 检索结果】                             │
│   参考资料：                                │
│   - 文档1...                                │
│   - 文档2...                                │
│                                             │
│  【4. 用户问题】                             │
│   用户问题：RAG 是什么？                    │
│                                             │
│  【5. 输出要求】                             │
│   "请用简洁的语言分点作答..."               │
│                                             │
│  【6. 约束条件】                             │
│   "如果参考资料中没有相关信息，请说明..."   │
│                                             │
└─────────────────────────────────────────────┘
```

### Prompt 设计原则

| 原则 | 说明 | 示例 |
|------|------|------|
| **明确性** | 清楚说明任务和要求 | "请根据参考资料回答，不要使用外部知识" |
| **结构性** | 有清晰的结构和层次 | 使用标题、列表、分隔符 |
| **完整性** | 包含所有必要信息 | 角色设定、任务说明、约束条件 |
| **简洁性** | 避免冗余信息 | 只保留关键的检索结果 |
| **可迭代** | 易于调整和优化 | 模块化设计，便于修改 |

### 常见的 Prompt 模板

**模板 1：基础问答型**

```
你是一个专业的问答助手。请根据以下参考资料回答用户问题。

参考资料：
{documents}

用户问题：{query}

请根据参考资料回答：
```

**模板 2：严格约束型**

```
你是一个严格的问答助手，必须基于提供的参考资料回答问题。

【重要规则】
1. 只使用参考资料中的信息
2. 如果参考资料中没有相关信息，请明确说"参考资料中未找到相关信息"
3. 不要编造、推测或使用外部知识
4. 回答要简洁准确

参考资料：
{documents}

用户问题：{query}

请根据以上规则回答：
```

**模板 3：结构化输出型**

```
你是一个专业的问答助手。请根据参考资料回答用户问题。

【回答要求】
1. 用分点形式回答
2. 每点不超过50字
3. 先给出结论，再展开说明
4. 标注信息来源

参考资料：
{documents}

用户问题：{query}

请按以下格式回答：
## 结论
...

## 详细说明
1. ...
   - 来源：文档X
2. ...
   - 来源：文档Y
```

### Prompt 优化技巧

| 技巧 | 说明 | 效果 |
|------|------|------|
| **Few-shot** | 提供示例问答 | 提升回答质量和格式一致性 |
| **CoT（思维链）** | 引导模型逐步推理 | 提升复杂问题的准确性 |
| **角色设定** | 设定专业角色 | 提升回答的专业性 |
| **约束条件** | 限制模型行为 | 减少幻觉，提升可靠性 |
| **输出格式** | 指定输出结构 | 提升可读性和可解析性 |

---

## 📦 示例代码

### 示例 1：基础 Prompt 构建

```python
# prompt_basic.py
# 基础 Prompt 构建示例

from typing import List


def build_basic_prompt(query: str, documents: List[str]) -> str:
    """
    构建基础 Prompt
    
    Args:
        query: 用户问题
        documents: 检索到的文档列表
    
    Returns:
        prompt: 构建好的 Prompt
    """
    # 拼接文档
    docs_text = "\n\n".join([f"文档{i+1}：{doc}" for i, doc in enumerate(documents)])
    
    # 构建 Prompt
    prompt = f"""你是一个专业的问答助手。请根据以下参考资料回答用户问题。

参考资料：
{docs_text}

用户问题：{query}

请根据参考资料回答："""
    
    return prompt


if __name__ == "__main__":
    query = "RAG 是什么？"
    
    documents = [
        "RAG（检索增强生成）是一种让大模型通过检索外部知识来增强回答的技术。",
        "RAG 的核心流程包括：文档加载、切分、向量化、检索、生成。",
        "RAG 适用于企业知识库、智能客服、文档问答等场景。"
    ]
    
    prompt = build_basic_prompt(query, documents)
    print(prompt)
```

---

### 示例 2：高级 Prompt 模板

```python
# prompt_templates.py
# 多种 Prompt 模板示例

from typing import List, Dict, Any, Optional


class PromptTemplate:
    """Prompt 模板类"""
    
    @staticmethod
    def basic(query: str, documents: List[str]) -> str:
        """基础模板"""
        docs_text = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
        
        return f"""请根据以下参考资料回答用户问题。

参考资料：
{docs_text}

用户问题：{query}

请回答："""
    
    @staticmethod
    def strict(query: str, documents: List[str]) -> str:
        """严格约束模板"""
        docs_text = "\n".join([f"【文档{i+1}】{doc}" for i, doc in enumerate(documents)])
        
        return f"""你是一个严格的问答助手，必须基于提供的参考资料回答问题。

【重要规则】
1. 只使用参考资料中的信息回答问题
2. 如果参考资料中没有相关信息，请明确说"参考资料中未找到相关信息"
3. 不要编造、推测或使用外部知识
4. 回答要简洁准确，不要添加无关内容

参考资料：
{docs_text}

用户问题：{query}

请根据以上规则回答："""
    
    @staticmethod
    def structured(query: str, documents: List[str]) -> str:
        """结构化输出模板"""
        docs_text = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
        
        return f"""你是一个专业的问答助手。请根据参考资料回答用户问题。

【回答要求】
1. 用分点形式回答（不超过5点）
2. 每点先说结论，再简要说明
3. 标注信息来源（如"来源：文档1"）
4. 如果有多个相关点，按重要性排序

参考资料：
{docs_text}

用户问题：{query}

请按以下格式回答：

## 核心结论
（一句话总结）

## 详细说明
1. ...
   - 来源：文档X
2. ...
   - 来源：文档Y
"""
    
    @staticmethod
    def few_shot(query: str, documents: List[str]) -> str:
        """Few-shot 模板（带示例）"""
        docs_text = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
        
        return f"""你是一个专业的问答助手。请根据参考资料回答用户问题。

【示例】
参考资料：
1. Python 是一门流行的编程语言，语法简洁。
2. Python 广泛应用于数据科学和 AI 开发。

用户问题：Python 有什么特点？

参考回答：
Python 的特点包括：
1. 语法简洁 - Python 的语法设计简洁优雅，易于学习
2. 应用广泛 - 广泛应用于数据科学和 AI 开发等领域

---

现在请根据以下参考资料回答：

参考资料：
{docs_text}

用户问题：{query}

请回答："""
    
    @staticmethod
    def with_context(
        query: str,
        documents: List[str],
        context: Dict[str, Any]
    ) -> str:
        """带上下文的模板"""
        docs_text = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
        
        # 提取上下文信息
        user_role = context.get("user_role", "普通用户")
        conversation_history = context.get("conversation_history", [])
        
        # 构建历史对话
        history_text = ""
        if conversation_history:
            history_text = "\n\n【历史对话】\n"
            for turn in conversation_history[-3:]:  # 只保留最近3轮
                history_text += f"用户：{turn['user']}\n"
                history_text += f"助手：{turn['assistant']}\n"
        
        return f"""你是一个专业的问答助手。当前用户角色：{user_role}。

{history_text}

【参考资料】
{docs_text}

用户问题：{query}

请根据参考资料回答，保持回答与历史对话的连贯性："""


def demo_templates():
    """演示不同的 Prompt 模板"""
    
    query = "RAG 有什么优势？"
    
    documents = [
        "RAG 可以避免大模型产生幻觉，提升回答的准确性。",
        "RAG 支持知识实时更新，无需重新训练模型。",
        "RAG 可以溯源回答来源，提升可解释性。"
    ]
    
    templates = {
        "基础模板": PromptTemplate.basic,
        "严格约束模板": PromptTemplate.strict,
        "结构化输出模板": PromptTemplate.structured,
        "Few-shot模板": PromptTemplate.few_shot
    }
    
    for name, template_func in templates.items():
        print("=" * 60)
        print(f"【{name}】")
        print("=" * 60)
        prompt = template_func(query, documents)
        print(prompt)
        print("\n")


if __name__ == "__main__":
    demo_templates()
```

---

### 示例 3：动态 Prompt 构建

```python
# prompt_dynamic.py
# 动态 Prompt 构建示例

from typing import List, Dict, Any, Optional
import json


class DynamicPromptBuilder:
    """动态 Prompt 构建器"""
    
    def __init__(
        self,
        system_prompt: str = "你是一个专业的问答助手。",
        max_docs: int = 5,
        max_doc_length: int = 500
    ):
        """
        初始化
        
        Args:
            system_prompt: 系统提示词
            max_docs: 最大文档数量
            max_doc_length: 单个文档最大长度
        """
        self.system_prompt = system_prompt
        self.max_docs = max_docs
        self.max_doc_length = max_doc_length
    
    def truncate_document(self, doc: str) -> str:
        """截断文档"""
        if len(doc) > self.max_doc_length:
            return doc[:self.max_doc_length] + "..."
        return doc
    
    def format_documents(
        self,
        documents: List[Dict[str, Any]],
        format_type: str = "numbered"
    ) -> str:
        """
        格式化文档
        
        Args:
            documents: 文档列表（包含 content 和 metadata）
            format_type: 格式类型
                - numbered: 编号列表
                - bullet: 无序列表
                - structured: 结构化（含元数据）
        
        Returns:
            formatted_docs: 格式化后的文档文本
        """
        # 截断文档
        truncated_docs = documents[:self.max_docs]
        
        if format_type == "numbered":
            return "\n".join([
                f"{i+1}. {self.truncate_document(doc['content'])}"
                for i, doc in enumerate(truncated_docs)
            ])
        
        elif format_type == "bullet":
            return "\n".join([
                f"- {self.truncate_document(doc['content'])}"
                for doc in truncated_docs
            ])
        
        elif format_type == "structured":
            lines = []
            for i, doc in enumerate(truncated_docs, 1):
                lines.append(f"【文档{i}】")
                lines.append(f"内容：{self.truncate_document(doc['content'])}")
                if 'metadata' in doc and doc['metadata']:
                    meta_str = ", ".join([f"{k}: {v}" for k, v in doc['metadata'].items()])
                    lines.append(f"来源：{meta_str}")
                lines.append("")
            return "\n".join(lines)
        
        else:
            return "\n".join([doc['content'] for doc in truncated_docs])
    
    def build(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        constraints: Optional[List[str]] = None,
        output_format: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        构建 Prompt
        
        Args:
            query: 用户问题
            documents: 检索到的文档
            constraints: 约束条件列表
            output_format: 输出格式要求
            conversation_history: 历史对话
            few_shot_examples: Few-shot 示例
        
        Returns:
            prompt: 构建好的 Prompt
        """
        parts = []
        
        # 1. 系统提示
        parts.append(f"【系统】\n{self.system_prompt}\n")
        
        # 2. Few-shot 示例
        if few_shot_examples:
            parts.append("【示例】")
            for example in few_shot_examples:
                parts.append(f"问题：{example['question']}")
                parts.append(f"回答：{example['answer']}")
            parts.append("")
        
        # 3. 历史对话
        if conversation_history:
            parts.append("【历史对话】")
            for turn in conversation_history[-3:]:
                parts.append(f"用户：{turn['user']}")
                parts.append(f"助手：{turn['assistant']}")
            parts.append("")
        
        # 4. 参考资料
        parts.append("【参考资料】")
        parts.append(self.format_documents(documents, format_type="structured"))
        
        # 5. 用户问题
        parts.append(f"【用户问题】\n{query}\n")
        
        # 6. 约束条件
        if constraints:
            parts.append("【重要约束】")
            for i, constraint in enumerate(constraints, 1):
                parts.append(f"{i}. {constraint}")
            parts.append("")
        
        # 7. 输出格式
        if output_format:
            parts.append(f"【输出格式】\n{output_format}\n")
        
        # 8. 最后的引导
        parts.append("请根据以上信息回答：")
        
        return "\n".join(parts)


def demo_dynamic_prompt():
    """演示动态 Prompt 构建"""
    
    # 准备数据
    query = "如何优化 RAG 系统的检索效果？"
    
    documents = [
        {
            "content": "使用混合检索（向量检索 + 关键词检索）可以提升检索效果。",
            "metadata": {"source": "技术文档", "page": 10}
        },
        {
            "content": "通过 Rerank 对检索结果进行二次排序，可以显著提升相关性。",
            "metadata": {"source": "技术文档", "page": 15}
        },
        {
            "content": "优化文档切分策略，选择合适的 chunk_size 和 overlap。",
            "metadata": {"source": "最佳实践", "page": 5}
        }
    ]
    
    constraints = [
        "只使用参考资料中的信息",
        "回答要简洁准确",
        "如果参考资料不足，请明确说明"
    ]
    
    output_format = """
请按以下格式回答：
## 核心建议
（1-2句话总结）

## 详细方法
1. ...
2. ...
"""
    
    # 构建 Prompt
    builder = DynamicPromptBuilder(
        system_prompt="你是一个 RAG 技术专家，擅长系统优化。",
        max_docs=3
    )
    
    prompt = builder.build(
        query=query,
        documents=documents,
        constraints=constraints,
        output_format=output_format
    )
    
    print("=" * 60)
    print("动态构建的 Prompt")
    print("=" * 60)
    print(prompt)


if __name__ == "__main__":
    demo_dynamic_prompt()
```

---

### 示例 4：完整 RAG Prompt 流程

```python
# prompt_rag_pipeline.py
# 完整的 RAG Prompt 构建流程

from typing import List, Dict, Any, Optional


class RAGPromptPipeline:
    """RAG Prompt 构建流水线"""
    
    def __init__(self):
        self.template_config = {
            "system_prompt": "你是一个专业的问答助手。",
            "max_context_length": 4000,
            "language": "zh"
        }
    
    def step1_filter_documents(
        self,
        documents: List[Dict[str, Any]],
        max_docs: int = 5,
        min_score: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        步骤1：过滤文档
        
        - 按分数过滤
        - 限制数量
        """
        filtered = [
            doc for doc in documents
            if doc.get('score', 0) >= min_score
        ]
        return filtered[:max_docs]
    
    def step2_deduplicate(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        步骤2：去重
        
        - 移除重复内容
        """
        seen = set()
        unique = []
        
        for doc in documents:
            content = doc.get('content', '')
            if content not in seen:
                seen.add(content)
                unique.append(doc)
        
        return unique
    
    def step3_format_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> str:
        """
        步骤3：格式化文档
        """
        formatted = []
        
        for i, doc in enumerate(documents, 1):
            content = doc.get('content', '无内容')
            source = doc.get('source', '未知来源')
            
            formatted.append(f"【参考资料{i}】（来源：{source}）")
            formatted.append(content)
            formatted.append("")
        
        return "\n".join(formatted)
    
    def step4_build_prompt(
        self,
        query: str,
        formatted_docs: str,
        style: str = "balanced"
    ) -> str:
        """
        步骤4：构建 Prompt
        
        Args:
            query: 用户问题
            formatted_docs: 格式化后的文档
            style: 风格
                - strict: 严格模式（只使用文档信息）
                - balanced: 平衡模式（允许适当推理）
                - creative: 创意模式（允许更多发挥）
        """
        if style == "strict":
            return f"""你是一个严格的问答助手。

【规则】
1. 只使用参考资料中的信息回答
2. 如果参考资料中没有相关信息，请说"参考资料中未找到相关信息"
3. 不要编造或推测任何内容
4. 回答要简洁准确

{formatted_docs}

用户问题：{query}

请根据规则回答："""
        
        elif style == "balanced":
            return f"""你是一个专业的问答助手。

请根据参考资料回答用户问题。如果参考资料信息不足，可以结合常识进行适当补充，但要明确标注。

{formatted_docs}

用户问题：{query}

请回答："""
        
        elif style == "creative":
            return f"""你是一个专业的问答助手。

请根据参考资料回答用户问题。可以结合参考资料和你的知识，给出全面详细的回答。

{formatted_docs}

用户问题：{query}

请详细回答："""
        
        else:
            return f"""请根据以下参考资料回答问题。

{formatted_docs}

用户问题：{query}

请回答："""
    
    def build(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        style: str = "balanced",
        max_docs: int = 5,
        min_score: float = 0.6
    ) -> str:
        """
        完整流程
        """
        # 步骤1：过滤
        filtered = self.step1_filter_documents(documents, max_docs, min_score)
        
        # 步骤2：去重
        unique = self.step2_deduplicate(filtered)
        
        # 步骤3：格式化
        formatted_docs = self.step3_format_documents(unique)
        
        # 步骤4：构建
        prompt = self.step4_build_prompt(query, formatted_docs, style)
        
        return prompt


def demo_rag_pipeline():
    """演示完整的 RAG Prompt 构建流程"""
    
    # 模拟检索结果
    documents = [
        {
            "content": "RAG 是检索增强生成技术。",
            "source": "基础知识",
            "score": 0.95
        },
        {
            "content": "RAG 结合了检索和生成两个阶段。",
            "source": "技术架构",
            "score": 0.88
        },
        {
            "content": "RAG 可以避免大模型幻觉问题。",
            "source": "优势介绍",
            "score": 0.82
        },
        {
            "content": "RAG 支持知识实时更新。",
            "source": "优势介绍",
            "score": 0.79
        },
        {
            "content": "这是一些无关内容，分数较低。",
            "source": "其他",
            "score": 0.45
        }
    ]
    
    query = "RAG 的主要优势是什么？"
    
    # 构建 Prompt
    pipeline = RAGPromptPipeline()
    
    # 测试不同风格
    styles = ["strict", "balanced", "creative"]
    
    for style in styles:
        print("=" * 60)
        print(f"【风格：{style}】")
        print("=" * 60)
        
        prompt = pipeline.build(
            query=query,
            documents=documents,
            style=style,
            max_docs=3,
            min_score=0.6
        )
        
        print(prompt)
        print("\n")


if __name__ == "__main__":
    demo_rag_pipeline()
```

---

## ⚠️ 常见坑

1. **Prompt 过长** — 检索文档太多导致 Prompt 超长，应该限制文档数量和长度
2. **缺少约束** — 没有限制大模型行为，导致幻觉严重
3. **格式不清晰** — Prompt 结构混乱，大模型难以理解
4. **忽略历史对话** — 多轮对话时没有传递上下文，回答不连贯
5. **Few-shot 示例不当** — 示例质量差，反而误导模型
6. **输出格式复杂** — 要求过于复杂的输出格式，模型难以遵守

---

## 🚀 实战建议

### 1. Prompt 模板选择指南

```
场景 → 推荐模板

简单问答
  → 基础模板

高准确性要求（医疗、法律）
  → 严格约束模板

需要结构化输出
  → 结构化输出模板

回答质量不稳定
  → Few-shot 模板

多轮对话
  → 带上下文的模板
```

### 2. 文档数量与长度建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 文档数量 | 3-5 个 | 太多会稀释关键信息 |
| 单文档长度 | 200-500 字符 | 截断过长文档 |
| Prompt 总长度 | < 4000 tokens | 避免超过模型限制 |

### 3. 优化技巧

| 技巧 | 适用场景 | 效果 |
|------|---------|------|
| 明确"不要做什么" | 幻觉严重 | 减少幻觉 |
| 提供示例 | 格式要求高 | 提升格式一致性 |
| 分段处理 | 复杂任务 | 提升准确性 |
| 角色设定 | 专业领域 | 提升专业性 |

---

## 📝 小练习

**代码练习：**

1. 运行 `prompt_basic.py`，理解基础 Prompt 构建

2. 运行 `prompt_templates.py`，对比不同模板的效果

3. **思考题**：
   - 如何平衡文档数量和 Prompt 长度？
   - Few-shot 示例的数量应该设置多少？
   - 如何评估 Prompt 的好坏？

4. **进阶练习**：实现一个 Prompt 优化器，根据回答质量自动调整 Prompt

---

## 📚 本章知识点汇总

| 知识点名称 | 知识点类型 | 知识点介绍 | 参考链接 |
|-----------|-----------|-----------|---------|
| Prompt Engineering | 技术 | 设计和优化输入给大模型的提示词，引导模型生成期望的回答 | [OpenAI Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering) |
| Few-shot Learning | 技术 | 在 Prompt 中提供少量示例，帮助模型理解任务要求 | [Few-shot Learning 论文](https://arxiv.org/abs/2005.14165) |
| CoT（Chain of Thought） | 技术 | 引导模型逐步推理，提升复杂问题的解决能力 | [CoT 论文](https://arxiv.org/abs/2201.11903) |
| 角色设定（Role Prompting） | 技术 | 在 Prompt 中设定模型的角色，提升回答的专业性 | - |
| 约束条件（Constraints） | 技术 | 在 Prompt 中明确限制模型的行为，减少幻觉 | - |
| 输出格式（Output Format） | 技术 | 指定模型输出的格式，提升可读性和可解析性 | - |
| 幻觉（Hallucination） | 概念 | 大模型生成看似正确但实际错误或虚构的内容 | - |
| 上下文窗口（Context Window） | 概念 | 模型一次能处理的最大 token 数量 | - |

---

## 📁 相关文件

| 文件名 | 说明 |
|--------|------|
| `README.md` | 本章学习笔记 |
| `prompt_basic.py` | 基础 Prompt 构建示例 |
| `prompt_templates.py` | 多种 Prompt 模板示例 |
| `prompt_dynamic.py` | 动态 Prompt 构建示例 |
| `prompt_rag_pipeline.py` | 完整 RAG Prompt 流程 |
