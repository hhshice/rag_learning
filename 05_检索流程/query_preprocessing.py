# -*- coding: utf-8 -*-
"""
Query 预处理示例
演示查询清洗、改写、扩展等技术
"""

import re
from typing import List, Dict


def clean_query(query: str) -> str:
    """
    清洗查询文本
    
    功能：
    - 去除多余空格
    - 去除特殊字符
    - 保留中文、英文、数字、常用标点
    
    Args:
        query: 原始查询文本
    
    Returns:
        cleaned_query: 清洗后的查询文本
    """
    # 去除首尾空格
    query = query.strip()
    
    # 去除多余空格（多个空格变为一个）
    query = re.sub(r'\s+', ' ', query)
    
    # 去除特殊字符（保留中文、英文、数字、常用标点）
    # \u4e00-\u9fff: 中文字符范围
    query = re.sub(r'[^\w\s\u4e00-\u9fff，。！？、；：""''（）【】]', '', query)
    
    return query


def expand_query(query: str, synonyms: Dict[str, List[str]]) -> str:
    """
    查询扩展：添加同义词和相关词
    
    原理：
    - 识别查询中的关键词
    - 添加同义词扩展查询范围
    - 提高召回率
    
    Args:
        query: 原始查询
        synonyms: 同义词字典，如 {"Python": ["python", "py", "Python3"]}
    
    Returns:
        expanded_query: 扩展后的查询
    """
    # 简单实现：在查询后添加同义词
    expanded_terms = []
    
    for word, syn_list in synonyms.items():
        if word in query:
            expanded_terms.extend(syn_list)
    
    if expanded_terms:
        # 去重并用空格连接
        unique_terms = list(set(expanded_terms))
        expanded_query = f"{query} {' '.join(unique_terms)}"
    else:
        expanded_query = query
    
    return expanded_query


def rewrite_query(query: str, context: str = None) -> str:
    """
    查询改写：补充上下文信息
    
    场景：
    - 用户问题模糊（如"这个怎么用"）
    - 需要结合上下文理解
    
    Args:
        query: 原始查询
        context: 上下文信息（如用户历史对话、当前场景）
    
    Returns:
        rewritten_query: 改写后的查询
    """
    # 定义模糊关键词
    vague_keywords = [
        "这个", "那个", "它", "它们",
        "怎么用", "怎么处理", "怎么办",
        "是什么", "怎么样"
    ]
    
    # 检查是否包含模糊词
    is_vague = any(keyword in query for keyword in vague_keywords)
    
    if is_vague and context:
        # 补充上下文
        rewritten_query = f"{context}：{query}"
    elif is_vague:
        # 无上下文时，添加引导词
        rewritten_query = f"请详细解释：{query}"
    else:
        rewritten_query = query
    
    return rewritten_query


def split_multi_intent_query(query: str) -> List[str]:
    """
    拆分多意图查询
    
    场景：
    - 一个查询包含多个问题
    - 需要拆分为子查询分别处理
    
    Args:
        query: 包含多个意图的查询
    
    Returns:
        sub_queries: 拆分后的子查询列表
    """
    # 定义分隔符（优先级从高到低）
    separators = ["？", "?", "，", ",", "；", ";", "和", "以及", "还有", "另外"]
    
    sub_queries = [query]
    
    for sep in separators:
        new_queries = []
        for q in sub_queries:
            # 按分隔符切分
            parts = q.split(sep)
            # 过滤空字符串并去除首尾空格
            parts = [p.strip() for p in parts if p.strip()]
            new_queries.extend(parts)
        sub_queries = new_queries
    
    # 过滤太短的查询（少于3个字符）
    sub_queries = [q for q in sub_queries if len(q) >= 3]
    
    return sub_queries


def correct_typos(query: str, typo_dict: Dict[str, str]) -> str:
    """
    纠正错别字
    
    Args:
        query: 原始查询
        typo_dict: 错别字纠正字典，如 {"Pythn": "Python", "javascrip": "javascript"}
    
    Returns:
        corrected_query: 纠正后的查询
    """
    corrected = query
    
    for typo, correct in typo_dict.items():
        corrected = corrected.replace(typo, correct)
    
    return corrected


def demo_query_preprocessing():
    """演示各种 Query 预处理技术"""
    
    print("=" * 60)
    print("🔧 Query 预处理演示")
    print("=" * 60)
    
    # 1. 清洗查询
    print("\n【1. 查询清洗】")
    raw_queries = [
        "  Python   是什么？ ！！！  ",
        "RAG@@@技术##介绍",
        "  如何学习  AI  "
    ]
    
    for raw in raw_queries:
        cleaned = clean_query(raw)
        print(f"  原始: '{raw}'")
        print(f"  清洗后: '{cleaned}'\n")
    
    # 2. 查询扩展
    print("\n【2. 查询扩展】")
    query = "Python 教程"
    synonyms = {
        "Python": ["python", "py", "Python3"],
        "教程": ["教程", "入门", "学习", "指南"]
    }
    
    expanded = expand_query(query, synonyms)
    print(f"  原始查询: '{query}'")
    print(f"  扩展后: '{expanded}'")
    
    # 3. 查询改写
    print("\n【3. 查询改写】")
    
    test_cases = [
        ("这个怎么用？", "FastAPI 框架"),
        ("它是什么？", None),
        ("Python 怎么样？", None)
    ]
    
    for query, ctx in test_cases:
        rewritten = rewrite_query(query, ctx)
        print(f"  原始: '{query}'")
        print(f"  上下文: {ctx if ctx else '无'}")
        print(f"  改写后: '{rewritten}'\n")
    
    # 4. 多意图拆分
    print("\n【4. 多意图拆分】")
    multi_intent_queries = [
        "Python 的优缺点是什么？怎么安装？",
        "RAG 是什么，和微调有什么区别",
        "如何学习 AI；推荐的学习资源有哪些？"
    ]
    
    for query in multi_intent_queries:
        sub_queries = split_multi_intent_query(query)
        print(f"  原始: '{query}'")
        print(f"  拆分后: {sub_queries}\n")
    
    # 5. 错别字纠正
    print("\n【5. 错别字纠正】")
    typo_dict = {
        "Pythn": "Python",
        "javascrip": "JavaScript",
        "RAGA": "RAG",
        "embeding": "embedding"
    }
    
    queries_with_typos = [
        "Pythn 怎么学",
        "javascrip 教程",
        "RAGA 技术介绍"
    ]
    
    for query in queries_with_typos:
        corrected = correct_typos(query, typo_dict)
        print(f"  原始: '{query}'")
        print(f"  纠正后: '{corrected}'\n")
    
    # 6. 完整预处理流程
    print("\n【6. 完整预处理流程】")
    
    def full_preprocessing(query: str, context: str = None) -> str:
        """完整的 Query 预处理流程"""
        # 步骤1: 清洗
        query = clean_query(query)
        
        # 步骤2: 纠正错别字
        query = correct_typos(query, typo_dict)
        
        # 步骤3: 改写
        query = rewrite_query(query, context)
        
        return query
    
    test_query = "  Pythn 怎么用？  "
    processed = full_preprocessing(test_query, context="Python 编程")
    
    print(f"  原始查询: '{test_query}'")
    print(f"  预处理后: '{processed}'")
    
    print("\n" + "=" * 60)
    print("✅ Query 预处理演示完成")
    print("=" * 60)
    print("\n💡 关键要点：")
    print("   1. 清洗：去除噪音，规范化格式")
    print("   2. 扩展：添加同义词，提高召回率")
    print("   3. 改写：补充上下文，消除歧义")
    print("   4. 拆分：多意图查询需要分别处理")
    print("   5. 纠错：修正错别字，避免检索失败")


if __name__ == "__main__":
    demo_query_preprocessing()
