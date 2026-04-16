# -*- coding: utf-8 -*-
"""
命令行 RAG 演示
提供交互式问答界面

运行方式：
python demo_cli.py
"""

from mini_rag import MiniRAG
import sys


def create_sample_knowledge_base():
    """创建示例知识库"""
    
    documents = [
        """
# RAG 技术介绍

RAG（检索增强生成）是一种让大模型通过检索外部知识来增强回答的技术。
它的核心思想是让大模型"开卷考试"，先从知识库中找到相关内容，再基于找到的内容生成回答。

RAG 的主要优势包括：
1. 避免幻觉：大模型基于真实文档回答，减少编造
2. 知识更新：无需重新训练模型，只需更新知识库
3. 可溯源：可以追踪回答的来源文档
4. 成本低：相比微调，RAG 的成本更低

RAG 适用于企业知识库、智能客服、文档问答等场景。
        """,
        """
# 向量数据库

向量数据库是专门用于存储和检索向量的数据库。它支持高效的相似度搜索，是 RAG 系统的核心组件。

常见的向量数据库包括：
- Chroma：轻量级，易于上手，适合中小项目
- FAISS：Facebook 开源，性能强大，适合本地开发
- Milvus：企业级，支持分布式，适合大规模应用
- Pinecone：云服务，免运维，适合快速上线

向量数据库的核心功能：
1. 向量存储：存储文本的向量表示
2. 相似度检索：快速找到最相似的向量
3. 元数据过滤：根据元数据条件过滤结果
        """,
        """
# Embedding 模型

Embedding 是将文本转换为向量的技术。通过 Embedding，文本的语义信息可以用数值表示，从而支持相似度计算。

常用的 Embedding 模型：
- all-MiniLM-L6-v2：速度快，效果好，适合入门
- text-embedding-ada-002：OpenAI 官方，效果好但需付费
- m3e-base：国产开源，中文效果好

Embedding 模型的选择会影响检索效果，需要根据具体场景选择。对于中文场景，推荐使用 m3e-base。
        """,
        """
# 文档切分策略

文档切分是 RAG 的关键步骤，直接影响检索效果。

常见的切分策略：
1. 固定长度切分：简单快速，但可能切断语义
2. 递归字符切分：按优先级尝试不同分隔符，语义较好（推荐）
3. 语义切分：基于 Embedding 相似度，语义最佳但成本高

推荐参数：
- chunk_size: 200-500 字符
- chunk_overlap: chunk_size 的 10-20%

对于不同类型的文档，需要调整切分参数以获得最佳效果。
        """,
        """
# Rerank 技术

Rerank 是对检索结果进行二次排序的技术，可以显著提升检索精度。

两阶段检索架构：
1. 召回阶段：向量检索召回 20-50 个文档（速度快但精度有限）
2. 精排阶段：Cross-Encoder 对召回文档精确排序（精度高但成本高）

Rerank 的优势：
- 提升相关性：Cross-Encoder 能捕捉 Query-Doc 深层交互
- 过滤噪音：通过分数阈值过滤低质量结果

推荐使用 Cross-Encoder 模型进行 Rerank，如 ms-marco-MiniLM-L-6-v2。
        """,
        """
# Prompt 设计

Prompt 设计是将检索结果和用户问题组织成合适格式，喂给大模型生成回答的过程。

Prompt 的核心组成部分：
1. 角色设定：定义模型的角色和专业性
2. 任务说明：明确说明任务目标
3. 参考资料：提供检索到的相关文档
4. 用户问题：用户的具体问题
5. 输出要求：指定输出格式和约束
6. 约束条件：限制模型行为，减少幻觉

好的 Prompt 能显著提升回答质量和准确性。
        """
    ]
    
    metadatas = [
        {"source": "RAG技术介绍", "category": "基础概念"},
        {"source": "向量数据库", "category": "技术组件"},
        {"source": "Embedding模型", "category": "技术组件"},
        {"source": "文档切分策略", "category": "最佳实践"},
        {"source": "Rerank技术", "category": "优化方法"},
        {"source": "Prompt设计", "category": "最佳实践"}
    ]
    
    return documents, metadatas


def print_banner():
    """打印横幅"""
    print("\n" + "=" * 60)
    print("🎯 最小 RAG Demo - 命令行版本")
    print("=" * 60)
    print("\n功能：")
    print("  - 文档切分与向量化")
    print("  - 语义检索与 Rerank")
    print("  - Prompt 自动构建")
    print("\n提示：")
    print("  - 输入问题进行查询")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 输入 'stats' 查看知识库统计")
    print("  - 输入 'clear' 清空知识库")
    print("=" * 60)


def print_welcome():
    """打印欢迎信息"""
    print("\n💡 这是一个最小 RAG 系统演示")
    print("   它可以帮你理解 RAG 的工作原理")
    print("   检索结果和构建的 Prompt 会显示在屏幕上")
    print("   你可以将 Prompt 发送给大模型（如 GPT-4）获得最终回答")


def main():
    """主函数"""
    
    # 打印横幅
    print_banner()
    
    # 初始化 RAG 系统
    print("\n🔄 初始化 RAG 系统...")
    rag = MiniRAG(
        persist_dir="./demo_chroma_db",
        collection_name="demo_kb",
        chunk_size=300,
        chunk_overlap=60
    )
    
    # 如果知识库为空，添加示例文档
    if rag.collection.count() == 0:
        print("\n📚 知识库为空，正在添加示例文档...")
        documents, metadatas = create_sample_knowledge_base()
        rag.add_documents(documents, metadatas)
    else:
        print(f"\nℹ️  知识库已有 {rag.collection.count()} 个文档")
    
    print_welcome()
    
    # 交互式问答
    print("\n" + "=" * 60)
    print("💬 开始问答")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n❓ 请输入问题：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 再见！")
            break
        
        # 退出命令
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 再见！")
            break
        
        # 查看统计
        if user_input.lower() == 'stats':
            stats = rag.get_stats()
            print("\n📊 知识库统计：")
            print(f"  - 总文档数：{stats['total_documents']}")
            print(f"  - 集合名称：{stats['collection_name']}")
            continue
        
        # 清空知识库
        if user_input.lower() == 'clear':
            confirm = input("⚠️  确定要清空知识库吗？(yes/no)：")
            if confirm.lower() == 'yes':
                rag.clear_collection()
            continue
        
        # 空输入
        if not user_input:
            continue
        
        # 执行查询
        try:
            result = rag.query(
                user_input,
                top_k=3,
                use_rerank=True,
                prompt_style="balanced",
                verbose=True
            )
            
            print("\n" + "=" * 60)
            print("💡 提示：将上面的 Prompt 发送给大模型（如 GPT-4）")
            print("   即可获得基于知识库的准确回答")
            print("=" * 60)
        
        except Exception as e:
            print(f"\n❌ 查询失败：{e}")
            print("   请检查输入或重试")


if __name__ == "__main__":
    main()
