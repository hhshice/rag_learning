# -*- coding: utf-8 -*-
"""
使用 Cohere Rerank API 进行重排序
Cohere 提供开箱即用的云端 Rerank 服务

安装依赖：pip install cohere

使用前需要：
1. 注册 Cohere 账号：https://dashboard.cohere.com/
2. 获取 API Key
3. 设置环境变量：export COHERE_API_KEY=your_api_key
"""

import os
from typing import List, Dict, Any, Optional


def cohere_rerank(
    query: str,
    documents: List[str],
    model: str = "rerank-multilingual-v2.0",
    top_n: int = 3,
    api_key: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    使用 Cohere Rerank API
    
    优点：
    - 开箱即用，无需加载模型
    - 支持多语言（包括中文）
    - 效果好，性能稳定
    - API 简单易用
    
    缺点：
    - 需要付费（有免费额度）
    - 依赖网络
    - 数据需上传到云端
    
    Args:
        query: 查询文本
        documents: 文档列表
        model: Rerank 模型名称
            - rerank-english-v2.0: 英文模型
            - rerank-multilingual-v2.0: 多语言模型（支持中文）
        top_n: 返回数量
        api_key: Cohere API Key（如未提供，从环境变量读取）
    
    Returns:
        results: 重排序结果列表
    """
    # 获取 API Key
    if api_key is None:
        api_key = os.environ.get("COHERE_API_KEY")
    
    if not api_key:
        print("⚠️  未找到 COHERE_API_KEY")
        print("   请设置环境变量或传入 api_key 参数")
        print("   获取 API Key: https://dashboard.cohere.com/")
        return None
    
    try:
        import cohere
        
        # 初始化客户端
        co = cohere.Client(api_key)
        
        # 调用 Rerank API
        response = co.rerank(
            query=query,
            documents=documents,
            model=model,
            top_n=top_n
        )
        
        # 整理结果
        results = []
        for result in response.results:
            results.append({
                "index": result.index,
                "document": documents[result.index],
                "relevance_score": result.relevance_score
            })
        
        return results
    
    except ImportError:
        print("❌ 未安装 cohere 库")
        print("   请运行：pip install cohere")
        return None
    
    except Exception as e:
        print(f"❌ Cohere API 调用失败: {e}")
        return None


def demo_cohere_rerank():
    """演示 Cohere Rerank"""
    
    print("=" * 60)
    print("🌐 Cohere Rerank API 演示")
    print("=" * 60)
    
    query = "如何学习 Python 编程"
    
    documents = [
        "Python 是一门流行的编程语言，语法简洁优雅。",
        "Python 安装教程：访问官网下载安装包并安装。",
        "机器学习是人工智能的一个分支，使用算法学习。",
        "Python 学习路线：从基础语法到进阶特性。",
        "深度学习框架 PyTorch 和 TensorFlow 对比。",
        "Python 编程实践：从零开始学编程。",
        "自然语言处理技术介绍。",
        "Python Web 开发框架 Django 和 FastAPI。"
    ]
    
    print(f"\n查询：{query}")
    print(f"文档数量：{len(documents)}")
    
    # 显示原始文档
    print("\n【原始文档】")
    print("-" * 60)
    for i, doc in enumerate(documents, 1):
        print(f"  [{i}] {doc}")
    
    # 调用 Cohere Rerank
    print("\n" + "=" * 60)
    print("🔄 调用 Cohere Rerank API...")
    print("=" * 60)
    
    results = cohere_rerank(
        query=query,
        documents=documents,
        model="rerank-multilingual-v2.0",
        top_n=3
    )
    
    if results:
        print("\n【Rerank 结果】")
        print("-" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"  [{i}] 相关性分数: {result['relevance_score']:.4f}")
            print(f"      原始索引: {result['index']}")
            print(f"      文档: {result['document']}\n")
        
        print("=" * 60)
        print("✅ Cohere Rerank 演示完成")
        print("=" * 60)
    
    else:
        print("\n" + "=" * 60)
        print("💡 提示：")
        print("   1. 注册 Cohere 账号：https://dashboard.cohere.com/")
        print("   2. 获取 API Key")
        print("   3. 设置环境变量：")
        print("      Windows: set COHERE_API_KEY=your_api_key")
        print("      Linux/Mac: export COHERE_API_KEY=your_api_key")
        print("=" * 60)


def compare_rerank_solutions():
    """对比不同的 Rerank 解决方案"""
    
    print("\n" + "=" * 60)
    print("📊 Rerank 解决方案对比")
    print("=" * 60)
    
    print("""
┌───────────────────────────────────────────────────────────────┐
│                    Rerank 解决方案对比                         │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  【1. Cross-Encoder（本地）】                                 │
│  ✅ 开源免费                                                  │
│  ✅ 数据隐私好（无需上传）                                    │
│  ✅ 效果好                                                    │
│  ❌ 需要下载模型（几百 MB）                                   │
│  ❌ 需要 GPU 加速（可选）                                     │
│  🎯 适用：中大型项目、对数据隐私有要求的场景                  │
│                                                               │
│  【2. Cohere Rerank API】                                     │
│  ✅ 开箱即用                                                  │
│  ✅ 支持多语言                                                │
│  ✅ 效果好且稳定                                              │
│  ❌ 需要付费（有免费额度）                                    │
│  ❌ 数据需上传到云端                                          │
│  🎯 适用：快速原型、中小型项目                                │
│                                                               │
│  【3. LLM Rerank（GPT-4）】                                   │
│  ✅ 效果最好                                                  │
│  ✅ 可解释性强                                                │
│  ❌ 成本极高                                                  │
│  ❌ 速度慢                                                    │
│  🎯 适用：高价值场景、对质量要求极高                          │
│                                                               │
└───────────────────────────────────────────────────────────────┘
    """)
    
    print("=" * 60)


def show_cohere_pricing():
    """显示 Cohere 的定价信息"""
    
    print("\n" + "=" * 60)
    print("💰 Cohere Rerank 定价")
    print("=" * 60)
    
    print("""
Cohere Rerank 定价（截至 2025 年）：

【免费额度】
  - 每月 1,000 次 Rerank 调用（适合测试和小型项目）

【付费计划】
  - 按使用量计费：$2 / 1,000 次 Rerank
  - 企业版：定制价格

【成本估算】
  - 小型项目（1万次/月）：$20/月
  - 中型项目（10万次/月）：$200/月
  - 大型项目（100万次/月）：$2,000/月

【官网】
  https://cohere.com/pricing

💡 提示：
  - 开发测试阶段可使用免费额度
  - 生产环境根据调用量选择方案
  - 也可考虑自建 Cross-Encoder（长期更省钱）
    """)
    
    print("=" * 60)


def demonstrate_without_api_key():
    """演示无 API Key 时的提示信息"""
    
    print("\n" + "=" * 60)
    print("📚 Cohere Rerank API 使用指南")
    print("=" * 60)
    
    print("""
本示例需要 Cohere API Key 才能运行。

【获取 API Key 步骤】

1. 注册 Cohere 账号
   访问：https://dashboard.cohere.com/signup

2. 创建 API Key
   登录后，在 API Keys 页面创建新的 Key

3. 设置环境变量
   
   Windows (PowerShell):
   $env:COHERE_API_KEY="your_api_key_here"
   
   Windows (CMD):
   set COHERE_API_KEY=your_api_key_here
   
   Linux/Mac:
   export COHERE_API_KEY=your_api_key_here

4. 或在代码中直接传入
   results = cohere_rerank(
       query="your query",
       documents=docs,
       api_key="your_api_key_here"
   )

【Cohere Rerank 优势】
  ✅ 无需下载模型
  ✅ 支持多语言（中文效果好）
  ✅ API 简单易用
  ✅ 效果稳定可靠

【示例代码】
  results = cohere_rerank(
      query="Python 如何安装",
      documents=["文档1", "文档2", ...],
      model="rerank-multilingual-v2.0",
      top_n=3
  )
    """)
    
    print("=" * 60)


if __name__ == "__main__":
    # 尝试运行演示
    demo_cohere_rerank()
    
    # 显示对比信息
    compare_rerank_solutions()
    
    # 显示定价信息
    show_cohere_pricing()
    
    # 如果没有 API Key，显示使用指南
    if not os.environ.get("COHERE_API_KEY"):
        demonstrate_without_api_key()
