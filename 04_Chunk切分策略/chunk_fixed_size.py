# -*- coding: utf-8 -*-
"""
固定长度切分示例
演示最简单的文档切分方法
"""


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


def main():
    """主函数"""
    # 测试文本
    text = """
RAG（检索增强生成）是一种让大模型通过检索外部知识来增强回答的技术。
它的核心流程包括：文档加载、文档切分、向量化、存储、检索、生成。
其中，文档切分是一个关键步骤，直接影响检索质量。
好的切分策略应该平衡语义完整性和检索精度。
切分太小会丢失上下文，切分太大会降低检索精度。
通常建议使用 200-500 字符作为 Chunk 大小，并设置 10-20% 的重叠。
"""
    
    # 执行切分
    chunks = fixed_size_chunk(text, chunk_size=50, overlap=10)
    
    print("=" * 60)
    print("📄 固定长度切分结果")
    print("=" * 60)
    print(f"原始文本长度: {len(text)} 字符")
    print(f"切分参数: chunk_size=50, overlap=10")
    print(f"切分后 Chunk 数量: {len(chunks)}")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} (长度: {len(chunk)}):")
        print(f"  {repr(chunk)}")
    
    print("\n" + "=" * 60)
    print("⚠️  固定长度切分的缺点：")
    print("   1. 可能切断句子，导致语义不完整")
    print("   2. 不考虑文档结构（标题、段落等）")
    print("   3. 推荐使用 LangChain 的递归字符切分器")
    print("=" * 60)


if __name__ == "__main__":
    main()
