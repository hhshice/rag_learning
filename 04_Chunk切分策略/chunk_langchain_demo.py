# -*- coding: utf-8 -*-
"""
使用 LangChain 进行文档切分（推荐方法）
演示递归字符切分器的使用

安装依赖：pip install langchain-text-splitters
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def demo_recursive_chunk():
    """
    演示 LangChain 的递归字符切分器
    
    递归字符切分器的工作原理：
    1. 尝试使用第一个分隔符（如 \n\n）切分
    2. 如果切分后的块仍然太大，使用下一个分隔符（如 \n）
    3. 依次尝试所有分隔符，直到满足大小要求
    """
    # 准备示例文本（模拟一篇文档）
    text = """
# 公司退换货政策

## 第一章 总则

本公司所有商品均支持7天无理由退换货。退换货政策适用于在官网购买的所有商品。

注意事项：
1. 商品需保持原包装完好
2. 需要提供购买凭证
3. 特殊商品（如内衣、食品）不支持退换

## 第二章 退货流程

### 2.1 申请退货

用户可在订单详情页面点击"申请退货"按钮，填写退货原因并提交申请。我们将在1-3个工作日内处理您的申请。

退货原因包括：
- 商品质量问题
- 商品与描述不符
- 个人原因（不喜欢、尺寸不合适等）

### 2.2 寄回商品

退货申请通过后，请在7天内将商品寄回。建议使用快递并保留快递单号，以便查询物流信息。

寄回地址：北京市朝阳区xxx路xxx号

## 第三章 退款说明

退款将在确认收到商品后的3-5个工作日内原路返回。

退款规则：
- 质量问题：全额退款，运费由公司承担
- 个人原因：全额退款，运费由用户承担
- 促销商品：按实际支付金额退款

如有疑问，请联系客服：400-xxx-xxxx
"""
    
    # 创建切分器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,          # 每个 Chunk 的最大字符数
        chunk_overlap=30,        # 相邻 Chunk 的重叠字符数
        length_function=len,     # 计算长度的函数
        separators=[
            "\n\n",              # 优先使用双换行（段落）
            "\n",                # 其次使用单换行
            "。",                # 中文句号
            "！",                # 中文感叹号
            "？",                # 中文问号
            "；",                # 中文分号
            " ",                 # 空格
            ""                   # 最后是单个字符
        ]
    )
    
    # 执行切分
    chunks = splitter.split_text(text)
    
    print("=" * 60)
    print("📄 递归字符切分结果（LangChain）")
    print("=" * 60)
    print(f"原始文本长度: {len(text)} 字符")
    print(f"切分参数: chunk_size=150, overlap=30")
    print(f"切分后 Chunk 数量: {len(chunks)}")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} (长度: {len(chunk)}) ---")
        print(chunk.strip())
    
    print("\n" + "=" * 60)
    print("✅ 递归字符切分的优点：")
    print("   1. 尽量保持段落完整性")
    print("   2. 优先按段落、句子切分，语义较好")
    print("   3. 灵活适应不同类型的文档")
    print("=" * 60)


def demo_chunk_with_metadata():
    """演示带元数据的切分"""
    from langchain_core.documents import Document
    
    text = """
# 产品说明书

## 产品介绍

本产品是一款智能家居设备，支持语音控制和远程操作。

## 使用方法

1. 首次使用需要连接 Wi-Fi
2. 下载手机 App 并注册账号
3. 按照提示完成设备绑定

## 注意事项

请勿在潮湿环境中使用本产品。
"""
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", " ", ""]
    )
    
    # 创建 Document 对象
    doc = Document(page_content=text, metadata={"source": "product_manual.txt"})
    
    # 切分文档（保留元数据）
    chunks = splitter.split_documents([doc])
    
    print("\n" + "=" * 60)
    print("📄 带元数据的切分结果")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"内容: {chunk.page_content.strip()}")
        print(f"元数据: {chunk.metadata}")


if __name__ == "__main__":
    demo_recursive_chunk()
    demo_chunk_with_metadata()
