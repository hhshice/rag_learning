# -*- coding: utf-8 -*-
"""
完整的 RAG Prompt 构建流程
演示从检索结果到最终 Prompt 的完整处理流程
"""

from typing import List, Dict, Any, Optional


class RAGPromptPipeline:
    """
    RAG Prompt 构建流水线
    
    完整流程：
    1. 过滤文档（按分数、数量）
    2. 去重文档
    3. 格式化文档
    4. 构建 Prompt
    """
    
    def __init__(
        self,
        system_prompt: str = "你是一个专业的问答助手。",
        max_context_tokens: int = 4000
    ):
        """
        初始化
        
        Args:
            system_prompt: 系统提示词
            max_context_tokens: 最大上下文 token 数
        """
        self.system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens
    
    def step1_filter_documents(
        self,
        documents: List[Dict[str, Any]],
        max_docs: int = 5,
        min_score: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        步骤1：过滤文档
        
        Args:
            documents: 原始文档列表
            max_docs: 最大文档数量
            min_score: 最低分数阈值
        
        Returns:
            filtered: 过滤后的文档列表
        """
        print(f"  [步骤1] 过滤文档")
        print(f"    - 原始文档数：{len(documents)}")
        
        # 按分数过滤
        filtered = [
            doc for doc in documents
            if doc.get('score', 0) >= min_score
        ]
        
        print(f"    - 按分数过滤后：{len(filtered)}（阈值：{min_score}）")
        
        # 限制数量
        filtered = filtered[:max_docs]
        
        print(f"    - 限制数量后：{len(filtered)}（最大：{max_docs}）")
        
        return filtered
    
    def step2_deduplicate(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        步骤2：去重
        
        移除内容重复的文档
        
        Args:
            documents: 文档列表
        
        Returns:
            unique: 去重后的文档列表
        """
        print(f"  [步骤2] 去重文档")
        print(f"    - 去重前：{len(documents)}")
        
        seen = set()
        unique = []
        
        for doc in documents:
            content = doc.get('content', '')
            
            # 使用内容的 hash 作为去重依据
            content_hash = hash(content)
            
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(doc)
        
        print(f"    - 去重后：{len(unique)}")
        
        return unique
    
    def step3_truncate_documents(
        self,
        documents: List[Dict[str, Any]],
        max_length: int = 500
    ) -> List[Dict[str, Any]]:
        """
        步骤3：截断文档
        
        限制每个文档的长度
        
        Args:
            documents: 文档列表
            max_length: 最大长度（字符）
        
        Returns:
            truncated: 截断后的文档列表
        """
        print(f"  [步骤3] 截断文档")
        print(f"    - 最大长度：{max_length} 字符")
        
        truncated = []
        total_length = 0
        
        for doc in documents:
            content = doc.get('content', '')
            
            if len(content) > max_length:
                content = content[:max_length] + "..."
                print(f"    - 文档已截断（原长度：{len(doc['content'])}）")
            
            truncated_doc = {**doc, 'content': content}
            truncated.append(truncated_doc)
            total_length += len(content)
        
        print(f"    - 总内容长度：{total_length} 字符")
        
        return truncated
    
    def step4_format_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> str:
        """
        步骤4：格式化文档
        
        将文档列表格式化为文本
        
        Args:
            documents: 文档列表
        
        Returns:
            formatted: 格式化后的文本
        """
        print(f"  [步骤4] 格式化文档")
        
        formatted = []
        
        for i, doc in enumerate(documents, 1):
            content = doc.get('content', '无内容')
            source = doc.get('source', '未知来源')
            score = doc.get('score', 0)
            
            formatted.append(f"【参考资料{i}】")
            formatted.append(f"来源：{source}（相关性：{score:.2f}）")
            formatted.append(f"内容：{content}")
            formatted.append("")  # 空行
        
        result = "\n".join(formatted)
        print(f"    - 格式化完成")
        
        return result
    
    def step5_build_prompt(
        self,
        query: str,
        formatted_docs: str,
        style: str = "balanced"
    ) -> str:
        """
        步骤5：构建 Prompt
        
        Args:
            query: 用户问题
            formatted_docs: 格式化后的文档
            style: 风格
                - strict: 严格模式
                - balanced: 平衡模式
                - creative: 创意模式
        
        Returns:
            prompt: 构建好的 Prompt
        """
        print(f"  [步骤5] 构建 Prompt（风格：{style}）")
        
        if style == "strict":
            prompt = f"""{self.system_prompt}

【重要规则】
1. 只使用参考资料中的信息回答
2. 如果参考资料中没有相关信息，请说"参考资料中未找到相关信息"
3. 不要编造或推测任何内容
4. 回答要简洁准确

{formatted_docs}

用户问题：{query}

请根据规则回答："""
        
        elif style == "balanced":
            prompt = f"""{self.system_prompt}

请根据参考资料回答用户问题。如果参考资料信息不足，可以结合常识补充，但要明确标注。

{formatted_docs}

用户问题：{query}

请回答："""
        
        elif style == "creative":
            prompt = f"""{self.system_prompt}

请根据参考资料回答用户问题。可以结合参考资料和你的知识，给出全面详细的回答。

{formatted_docs}

用户问题：{query}

请详细回答："""
        
        else:
            prompt = f"""请根据以下参考资料回答问题。

{formatted_docs}

用户问题：{query}

请回答："""
        
        print(f"    - Prompt 长度：{len(prompt)} 字符")
        
        return prompt
    
    def build(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        style: str = "balanced",
        max_docs: int = 5,
        min_score: float = 0.6,
        max_doc_length: int = 500
    ) -> str:
        """
        完整构建流程
        
        Args:
            query: 用户问题
            documents: 检索结果
            style: Prompt 风格
            max_docs: 最大文档数
            min_score: 最低分数
            max_doc_length: 最大文档长度
        
        Returns:
            prompt: 最终的 Prompt
        """
        print("=" * 60)
        print("🔄 RAG Prompt 构建流程")
        print("=" * 60)
        
        # 步骤1：过滤
        filtered = self.step1_filter_documents(documents, max_docs, min_score)
        
        # 步骤2：去重
        unique = self.step2_deduplicate(filtered)
        
        # 步骤3：截断
        truncated = self.step3_truncate_documents(unique, max_doc_length)
        
        # 步骤4：格式化
        formatted = self.step4_format_documents(truncated)
        
        # 步骤5：构建
        prompt = self.step5_build_prompt(query, formatted, style)
        
        print("\n" + "=" * 60)
        print("✅ Prompt 构建完成")
        print("=" * 60)
        
        return prompt


def demo_full_pipeline():
    """演示完整的构建流程"""
    
    # 模拟检索结果
    documents = [
        {
            "content": "RAG 是检索增强生成技术，结合了检索和生成两个阶段。",
            "source": "基础知识",
            "score": 0.95
        },
        {
            "content": "RAG 的核心流程包括：文档加载、切分、向量化、检索、生成。",
            "source": "技术架构",
            "score": 0.88
        },
        {
            "content": "RAG 可以避免大模型幻觉问题，提升回答的准确性。",
            "source": "优势介绍",
            "score": 0.82
        },
        {
            "content": "RAG 支持知识实时更新，无需重新训练模型。",
            "source": "优势介绍",
            "score": 0.79
        },
        {
            "content": "RAG 适用于企业知识库、智能客服等场景。",
            "source": "应用场景",
            "score": 0.75
        },
        {
            "content": "这是一些无关内容，分数较低。",
            "source": "其他",
            "score": 0.45
        },
        {
            "content": "RAG 可以避免大模型幻觉问题，提升回答的准确性。",  # 重复内容
            "source": "优势介绍",
            "score": 0.80
        }
    ]
    
    query = "RAG 的主要优势是什么？"
    
    # 构建 Prompt
    pipeline = RAGPromptPipeline(
        system_prompt="你是一个 RAG 技术专家。"
    )
    
    prompt = pipeline.build(
        query=query,
        documents=documents,
        style="strict",
        max_docs=5,
        min_score=0.6,
        max_doc_length=200
    )
    
    print("\n" + "=" * 60)
    print("最终 Prompt：")
    print("=" * 60)
    print(prompt)


def compare_styles():
    """对比不同风格"""
    
    print("\n" + "=" * 60)
    print("📊 不同风格对比")
    print("=" * 60)
    
    documents = [
        {
            "content": "Python 是一门流行的编程语言。",
            "source": "基础",
            "score": 0.9
        },
        {
            "content": "Python 广泛应用于 AI 开发。",
            "source": "应用",
            "score": 0.85
        }
    ]
    
    query = "Python 是什么？"
    
    pipeline = RAGPromptPipeline()
    
    styles = ["strict", "balanced", "creative"]
    
    for style in styles:
        print(f"\n【风格：{style}】")
        print("-" * 60)
        
        prompt = pipeline.build(
            query=query,
            documents=documents,
            style=style,
            max_docs=3,
            min_score=0.6
        )
        
        # 只打印部分内容
        print(prompt[:300] + "...")
    
    print("\n" + "=" * 60)
    print("💡 选择建议：")
    print("   - strict: 高准确性要求（医疗、法律）")
    print("   - balanced: 一般问答场景")
    print("   - creative: 需要创造性回答的场景")
    print("=" * 60)


if __name__ == "__main__":
    demo_full_pipeline()
    compare_styles()
