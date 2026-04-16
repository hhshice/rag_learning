# -*- coding: utf-8 -*-
"""
多种 Prompt 模板示例
演示不同场景下的 Prompt 设计策略
"""

from typing import List, Dict, Any


class PromptTemplate:
    """Prompt 模板集合"""
    
    @staticmethod
    def basic(query: str, documents: List[str]) -> str:
        """
        基础模板
        
        特点：
        - 简洁明了
        - 适用于简单问答场景
        """
        docs_text = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
        
        return f"""你是一个专业的问答助手。请根据以下参考资料回答用户问题。

参考资料：
{docs_text}

用户问题：{query}

请根据参考资料回答："""
    
    @staticmethod
    def strict(query: str, documents: List[str]) -> str:
        """
        严格约束模板
        
        特点：
        - 明确禁止编造信息
        - 强调只使用参考资料
        - 适用于高准确性场景（医疗、法律）
        """
        docs_text = "\n".join([f"【文档{i+1}】{doc}" for i, doc in enumerate(documents)])
        
        return f"""你是一个严格的问答助手，必须基于提供的参考资料回答问题。

【重要规则】
1. 只使用参考资料中的信息回答问题
2. 如果参考资料中没有相关信息，请明确说"参考资料中未找到相关信息"
3. 不要编造、推测或使用外部知识
4. 回答要简洁准确，不要添加无关内容
5. 如果不确定，请如实说明

参考资料：
{docs_text}

用户问题：{query}

请根据以上规则回答："""
    
    @staticmethod
    def structured(query: str, documents: List[str]) -> str:
        """
        结构化输出模板
        
        特点：
        - 指定输出格式
        - 要求标注来源
        - 适用于需要结构化输出的场景
        """
        docs_text = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
        
        return f"""你是一个专业的问答助手。请根据参考资料回答用户问题。

【回答要求】
1. 用分点形式回答（不超过5点）
2. 每点先说结论，再简要说明
3. 标注信息来源（如"来源：文档1"）
4. 如果有多个相关点，按重要性排序
5. 控制总字数在200字以内

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
        """
        Few-shot 模板（带示例）
        
        特点：
        - 提供示例问答
        - 帮助模型理解任务
        - 提升回答质量和格式一致性
        """
        docs_text = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
        
        return f"""你是一个专业的问答助手。请根据参考资料回答用户问题。

【示例】

参考资料：
1. Python 是一门流行的编程语言，语法简洁。
2. Python 广泛应用于数据科学和 AI 开发。

用户问题：Python 有什么特点？

参考回答：
Python 的特点包括：
1. 语法简洁 - Python 的语法设计简洁优雅，易于学习和使用
   - 来源：文档1
2. 应用广泛 - 广泛应用于数据科学和 AI 开发等领域
   - 来源：文档2

---

现在请根据以下参考资料回答：

参考资料：
{docs_text}

用户问题：{query}

请按示例格式回答："""
    
    @staticmethod
    def conversation(
        query: str,
        documents: List[str],
        history: List[Dict[str, str]]
    ) -> str:
        """
        多轮对话模板
        
        特点：
        - 包含历史对话
        - 保持回答连贯性
        - 适用于对话式问答场景
        """
        docs_text = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
        
        # 构建历史对话
        history_text = ""
        if history:
            history_text = "\n\n【历史对话】\n"
            for turn in history[-3:]:  # 只保留最近3轮
                history_text += f"用户：{turn['user']}\n"
                history_text += f"助手：{turn['assistant']}\n"
        
        return f"""你是一个专业的问答助手。

{history_text}

【参考资料】
{docs_text}

用户问题：{query}

请根据参考资料回答，保持与历史对话的连贯性："""
    
    @staticmethod
    def role_based(
        query: str,
        documents: List[str],
        role: str = "技术专家"
    ) -> str:
        """
        角色设定模板
        
        特点：
        - 设定专业角色
        - 提升回答专业性
        - 适用于特定领域
        """
        docs_text = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
        
        role_prompts = {
            "技术专家": "你是一个资深技术专家，擅长用通俗易懂的方式解释复杂概念。",
            "教师": "你是一个耐心的教师，善于用简单的语言讲解知识。",
            "客服": "你是一个专业的客服，态度友好，回答简洁准确。",
            "医生": "你是一个专业的医生，回答严谨准确，强调参考专业意见。"
        }
        
        role_prompt = role_prompts.get(role, role_prompts["技术专家"])
        
        return f"""{role_prompt}

请根据以下参考资料回答用户问题。

参考资料：
{docs_text}

用户问题：{query}

请回答："""


def demo_all_templates():
    """演示所有模板"""
    
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
        "Few-shot模板": PromptTemplate.few_shot,
    }
    
    for name, template_func in templates.items():
        print("=" * 60)
        print(f"【{name}】")
        print("=" * 60)
        prompt = template_func(query, documents)
        print(prompt)
        print("\n")
    
    # 演示多轮对话模板
    print("=" * 60)
    print("【多轮对话模板】")
    print("=" * 60)
    
    history = [
        {"user": "什么是 RAG？", "assistant": "RAG 是检索增强生成技术..."},
        {"user": "它的工作原理是什么？", "assistant": "RAG 的工作原理包括..."}
    ]
    
    prompt = PromptTemplate.conversation(query, documents, history)
    print(prompt)
    print("\n")
    
    # 演示角色设定模板
    print("=" * 60)
    print("【角色设定模板 - 技术专家】")
    print("=" * 60)
    
    prompt = PromptTemplate.role_based(query, documents, role="技术专家")
    print(prompt)


def compare_template_effects():
    """对比不同模板的效果差异"""
    
    print("\n" + "=" * 60)
    print("📊 模板效果对比")
    print("=" * 60)
    
    query = "Python 如何安装？"
    
    documents = [
        "Python 安装教程：访问官网 python.org 下载对应系统的安装包。",
        "安装完成后，需要配置环境变量。"
    ]
    
    print(f"\n用户问题：{query}")
    print(f"文档数量：{len(documents)}\n")
    
    # 基础模板
    print("【基础模板】")
    print("-" * 60)
    print("特点：简洁，依赖模型自身理解")
    print("适用：简单问答\n")
    
    # 严格约束模板
    print("【严格约束模板】")
    print("-" * 60)
    print("特点：明确限制，减少幻觉")
    print("适用：高准确性场景\n")
    
    # 结构化模板
    print("【结构化输出模板】")
    print("-" * 60)
    print("特点：格式清晰，易于解析")
    print("适用：需要结构化输出\n")
    
    print("=" * 60)
    print("💡 选择建议：")
    print("   - 简单问答 → 基础模板")
    print("   - 高准确性 → 严格约束模板")
    print("   - 结构化输出 → 结构化模板")
    print("   - 质量不稳定 → Few-shot 模板")
    print("=" * 60)


if __name__ == "__main__":
    demo_all_templates()
    compare_template_effects()
