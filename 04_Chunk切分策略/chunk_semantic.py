# -*- coding: utf-8 -*-
"""
语义切分示例 - 基于 Embedding 相似度
高级切分方法，语义最佳但计算成本较高

安装依赖：pip install sentence-transformers numpy
"""

from sentence_transformers import SentenceTransformer
import numpy as np


def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    
    Args:
        vec1: 向量1
        vec2: 向量2
    
    Returns:
        相似度值，范围 [-1, 1]，1 表示完全相同
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def semantic_chunk(text, model_name='all-MiniLM-L6-v2', threshold=0.5):
    """
    基于语义相似度的文档切分
    
    原理：
    1. 将文本按句子切分
    2. 计算相邻句子的相似度
    3. 相似度低的句子之间作为切分点
    
    Args:
        text: 原始文本
        model_name: Embedding 模型名称
        threshold: 相似度阈值，低于此值则切分
    
    Returns:
        chunks: 切分后的 Chunk 列表
    """
    # 加载模型
    print(f"🔄 加载 Embedding 模型: {model_name}")
    model = SentenceTransformer(model_name)
    
    # 按句子切分（简单实现，实际可用更复杂的分句逻辑）
    # 这里使用中文句号作为分隔符
    sentences = [s.strip() for s in text.replace('\n', ' ').split('。') if s.strip()]
    
    if len(sentences) == 0:
        return [text]
    
    if len(sentences) == 1:
        return [text]
    
    print(f"📄 分句完成: {len(sentences)} 个句子")
    
    # 生成句子向量
    print("🔄 生成句子向量...")
    embeddings = model.encode(sentences, show_progress_bar=False)
    
    # 计算相邻句子的相似度
    print("🔄 计算句子相似度...")
    similarities = []
    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i-1], embeddings[i])
        similarities.append(sim)
    
    # 根据相似度切分
    chunks = []
    current_chunk = [sentences[0]]
    
    for i, sim in enumerate(similarities):
        if sim < threshold:
            # 相似度低，在此切分
            chunks.append('。'.join(current_chunk) + '。')
            current_chunk = [sentences[i+1]]
        else:
            # 相似度高，继续合并
            current_chunk.append(sentences[i+1])
    
    # 添加最后一个 Chunk
    if current_chunk:
        chunks.append('。'.join(current_chunk) + '。')
    
    return chunks, similarities


def demo_semantic_chunk():
    """演示语义切分"""
    # 测试文本（包含不同主题的段落）
    text = """
机器学习是人工智能的一个分支。它使用统计技术让计算机系统能够从数据中学习。
深度学习是机器学习的子领域。它使用多层神经网络来学习数据的层次化表示。
自然语言处理是AI的重要应用领域。它让计算机能够理解和生成人类语言。
计算机视觉让机器能够看懂图像和视频。它在自动驾驶、医疗影像等领域有广泛应用。
强化学习通过试错来学习策略。它在游戏AI、机器人控制等场景表现出色。
"""
    
    print("=" * 60)
    print("🧠 语义切分演示")
    print("=" * 60)
    
    # 执行语义切分
    chunks, similarities = semantic_chunk(text, threshold=0.5)
    
    print(f"\n原始文本长度: {len(text)} 字符")
    print(f"切分后 Chunk 数量: {len(chunks)}")
    print("=" * 60)
    
    # 显示相似度信息
    print("\n相邻句子相似度：")
    for i, sim in enumerate(similarities):
        status = "✂️ 切分" if sim < 0.5 else "➡️ 合并"
        print(f"  句子 {i} → {i+1}: {sim:.4f} {status}")
    
    # 显示切分结果
    print("\n" + "=" * 60)
    print("📄 切分结果：")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} (长度: {len(chunk)}) ---")
        print(chunk.strip())
    
    print("\n" + "=" * 60)
    print("💡 语义切分的特点：")
    print("   ✅ 优点：语义最佳，Chunk 内容高度相关")
    print("   ❌ 缺点：需要计算 Embedding，成本较高")
    print("   🎯 适用：高精度需求场景")
    print("=" * 60)


def compare_chunk_methods():
    """对比不同切分方法"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    text = """
机器学习是人工智能的一个分支。它使用统计技术让计算机系统能够从数据中学习。
深度学习是机器学习的子领域。它使用多层神经网络来学习数据的层次化表示。
自然语言处理是AI的重要应用领域。它让计算机能够理解和生成人类语言。
"""
    
    print("\n" + "=" * 60)
    print("📊 切分方法对比")
    print("=" * 60)
    
    # 方法1：固定长度切分
    print("\n【方法1：固定长度切分】")
    chunk_size = 50
    start = 0
    chunks1 = []
    while start < len(text):
        chunks1.append(text[start:start+chunk_size])
        start += chunk_size
    print(f"Chunk 数量: {len(chunks1)}")
    for i, chunk in enumerate(chunks1[:3], 1):  # 只显示前3个
        print(f"  Chunk {i}: {repr(chunk[:30])}...")
    
    # 方法2：递归字符切分
    print("\n【方法2：递归字符切分】")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
        separators=["\n\n", "\n", "。", " ", ""]
    )
    chunks2 = splitter.split_text(text)
    print(f"Chunk 数量: {len(chunks2)}")
    for i, chunk in enumerate(chunks2[:3], 1):
        print(f"  Chunk {i}: {chunk[:30]}...")
    
    # 方法3：语义切分
    print("\n【方法3：语义切分】")
    chunks3, _ = semantic_chunk(text, threshold=0.5)
    print(f"Chunk 数量: {len(chunks3)}")
    for i, chunk in enumerate(chunks3[:3], 1):
        print(f"  Chunk {i}: {chunk[:30]}...")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_semantic_chunk()
    compare_chunk_methods()
