# -*- coding: utf-8 -*-
"""
基础 Prompt 构建示例
演示如何将检索结果和用户问题组织成 Prompt
"""

from typing import List


def build_basic_prompt(query: str, documents: List[str]) -> str:
    """
    构建基础 Prompt
    
    这是最简单的 Prompt 构建方式：
    1. 拼接所有检索到的文档
    2. 加上用户问题
    3. 引导模型回答
    
    Args:
        query: 用户问题
        documents: 检索到的文档列表
    
    Returns:
        prompt: 构建好的 Prompt
    """
    # 拼接文档
    docs_text = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
    
    # 构建 Prompt
    prompt = f"""你是一个专业的问答助手。请根据以下参考资料回答用户问题。

参考资料：
{docs_text}

用户问题：{query}

请根据参考资料回答："""
    
    return prompt


def demonstrate_basic_prompt():
    """演示基础 Prompt 构建"""
    
    print("=" * 60)
    print("📝 基础 Prompt 构建")
    print("=" * 60)
    
    # 用户问题
    query = "RAG 是什么？有什么优势？"
    
    # 检索到的文档（模拟）
    documents = [
        "RAG（检索增强生成）是一种让大模型通过检索外部知识来增强回答的技术。",
        "RAG 的核心流程包括：文档加载、切分、向量化、检索、生成。",
        "RAG 可以避免大模型产生幻觉，提升回答的准确性。",
        "RAG 支持知识实时更新，无需重新训练模型。",
        "RAG 适用于企业知识库、智能客服、文档问答等场景。"
    ]
    
    print(f"\n用户问题：{query}")
    print(f"检索到的文档数量：{len(documents)}")
    
    # 构建 Prompt
    prompt = build_basic_prompt(query, documents)
    
    print("\n" + "=" * 60)
    print("构建的 Prompt：")
    print("=" * 60)
    print(prompt)
    
    print("\n" + "=" * 60)
    print("💡 提示：")
    print("   - 这是基础的 Prompt 构建方式")
    print("   - 适用于简单的问答场景")
    print("   - 可以根据需求添加更多约束和格式要求")
    print("=" * 60)


def compare_with_without_context():
    """对比有无上下文的差异"""
    
    print("\n" + "=" * 60)
    print("📊 对比：有无参考资料")
    print("=" * 60)
    
    query = "RAG 的主要优势是什么？"
    
    # 无参考资料
    print("\n【无参考资料】")
    print("-" * 60)
    prompt_no_context = f"""请回答用户问题。

用户问题：{query}

请回答："""
    print(prompt_no_context)
    print("\n可能的问题：模型可能编造信息，产生幻觉")
    
    # 有参考资料
    print("\n【有参考资料】")
    print("-" * 60)
    
    documents = [
        "RAG 可以避免大模型产生幻觉，提升回答的准确性。",
        "RAG 支持知识实时更新，无需重新训练模型。"
    ]
    
    prompt_with_context = build_basic_prompt(query, documents)
    print(prompt_with_context)
    print("\n优势：模型基于真实信息回答，减少幻觉")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demonstrate_basic_prompt()
    compare_with_without_context()
