# -*- coding: utf-8 -*-
"""
动态 Prompt 构建示例
演示如何根据不同场景动态构建 Prompt
"""

from typing import List, Dict, Any, Optional


class DynamicPromptBuilder:
    """
    动态 Prompt 构建器
    
    特点：
    - 可配置的参数
    - 智能文档处理
    - 多种输出格式
    - 支持上下文
    """
    
    def __init__(
        self,
        system_prompt: str = "你是一个专业的问答助手。",
        max_docs: int = 5,
        max_doc_length: int = 500,
        language: str = "zh"
    ):
        """
        初始化
        
        Args:
            system_prompt: 系统提示词
            max_docs: 最大文档数量
            max_doc_length: 单个文档最大长度（字符）
            language: 语言（zh/en）
        """
        self.system_prompt = system_prompt
        self.max_docs = max_docs
        self.max_doc_length = max_doc_length
        self.language = language
    
    def truncate_document(self, doc: str) -> str:
        """
        截断文档
        
        如果文档过长，截断并添加省略号
        """
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
        # 限制文档数量
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
                
                # 添加元数据
                if 'metadata' in doc and doc['metadata']:
                    meta_items = []
                    for k, v in doc['metadata'].items():
                        meta_items.append(f"{k}: {v}")
                    lines.append(f"信息：{', '.join(meta_items)}")
                
                lines.append("")  # 空行
            
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
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        doc_format: str = "numbered"
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
            doc_format: 文档格式
        
        Returns:
            prompt: 构建好的 Prompt
        """
        parts = []
        
        # 1. 系统提示
        parts.append(f"【系统】\n{self.system_prompt}\n")
        
        # 2. Few-shot 示例（可选）
        if few_shot_examples:
            parts.append("【示例】")
            for i, example in enumerate(few_shot_examples, 1):
                parts.append(f"\n示例{i}：")
                parts.append(f"问题：{example['question']}")
                parts.append(f"回答：{example['answer']}")
            parts.append("")
        
        # 3. 历史对话（可选）
        if conversation_history:
            parts.append("【历史对话】")
            for turn in conversation_history[-3:]:  # 最近3轮
                parts.append(f"用户：{turn['user']}")
                parts.append(f"助手：{turn['assistant']}")
            parts.append("")
        
        # 4. 参考资料
        parts.append("【参考资料】")
        parts.append(self.format_documents(documents, format_type=doc_format))
        
        # 5. 用户问题
        parts.append(f"\n【用户问题】\n{query}\n")
        
        # 6. 约束条件（可选）
        if constraints:
            parts.append("【重要约束】")
            for i, constraint in enumerate(constraints, 1):
                parts.append(f"{i}. {constraint}")
            parts.append("")
        
        # 7. 输出格式（可选）
        if output_format:
            parts.append(f"【输出格式】\n{output_format}\n")
        
        # 8. 最后的引导
        parts.append("请根据以上信息回答：")
        
        return "\n".join(parts)


def demo_dynamic_builder():
    """演示动态 Prompt 构建"""
    
    print("=" * 60)
    print("🔧 动态 Prompt 构建演示")
    print("=" * 60)
    
    # 准备数据
    query = "如何优化 RAG 系统的检索效果？"
    
    documents = [
        {
            "content": "使用混合检索（向量检索 + 关键词检索）可以提升检索效果。",
            "metadata": {"source": "技术文档", "page": 10}
        },
        {
            "content": "通过 Rerank 对检索结果进行二次排序，可以显著提升相关性。Cross-Encoder 是常用的 Rerank 模型。",
            "metadata": {"source": "技术文档", "page": 15}
        },
        {
            "content": "优化文档切分策略，选择合适的 chunk_size 和 overlap 参数。",
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
        max_docs=3,
        max_doc_length=200
    )
    
    prompt = builder.build(
        query=query,
        documents=documents,
        constraints=constraints,
        output_format=output_format,
        doc_format="structured"
    )
    
    print(prompt)
    
    print("\n" + "=" * 60)
    print("✅ Prompt 构建完成")
    print("=" * 60)


def demo_with_few_shot():
    """演示带 Few-shot 的 Prompt"""
    
    print("\n" + "=" * 60)
    print("📚 Few-shot Prompt 演示")
    print("=" * 60)
    
    query = "向量数据库的作用是什么？"
    
    documents = [
        {"content": "向量数据库用于存储和检索文本的向量表示。"},
        {"content": "向量数据库支持高效的相似度搜索。"}
    ]
    
    few_shot_examples = [
        {
            "question": "Python 是什么？",
            "answer": "Python 是一门流行的编程语言。\n特点：语法简洁，应用广泛。"
        }
    ]
    
    builder = DynamicPromptBuilder()
    
    prompt = builder.build(
        query=query,
        documents=documents,
        few_shot_examples=few_shot_examples,
        doc_format="bullet"
    )
    
    print(prompt)
    print("\n" + "=" * 60)


def demo_with_conversation():
    """演示带历史对话的 Prompt"""
    
    print("\n" + "=" * 60)
    print("💬 多轮对话 Prompt 演示")
    print("=" * 60)
    
    query = "它有哪些优势？"
    
    documents = [
        {"content": "RAG 可以避免大模型幻觉问题。"},
        {"content": "RAG 支持知识实时更新。"}
    ]
    
    history = [
        {
            "user": "什么是 RAG？",
            "assistant": "RAG 是检索增强生成技术，让大模型通过检索外部知识来增强回答。"
        },
        {
            "user": "它的工作原理是什么？",
            "assistant": "RAG 的工作原理包括：文档加载、切分、向量化、检索、生成等步骤。"
        }
    ]
    
    builder = DynamicPromptBuilder()
    
    prompt = builder.build(
        query=query,
        documents=documents,
        conversation_history=history,
        doc_format="numbered"
    )
    
    print(prompt)
    print("\n" + "=" * 60)


def demo_different_doc_formats():
    """演示不同的文档格式"""
    
    print("\n" + "=" * 60)
    print("📄 文档格式对比")
    print("=" * 60)
    
    query = "什么是向量数据库？"
    
    documents = [
        {"content": "向量数据库是存储向量的数据库。", "metadata": {"source": "基础"}},
        {"content": "支持高效的相似度搜索。", "metadata": {"source": "特性"}}
    ]
    
    builder = DynamicPromptBuilder(max_docs=2)
    
    formats = ["numbered", "bullet", "structured"]
    
    for fmt in formats:
        print(f"\n【格式：{fmt}】")
        print("-" * 60)
        
        prompt = builder.build(
            query=query,
            documents=documents,
            doc_format=fmt
        )
        
        print(prompt)


if __name__ == "__main__":
    demo_dynamic_builder()
    demo_with_few_shot()
    demo_with_conversation()
    demo_different_doc_formats()
