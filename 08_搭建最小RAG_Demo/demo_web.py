# -*- coding: utf-8 -*-
"""
Web API 版本 - FastAPI
提供 RESTful API 接口

运行方式：
python demo_web.py

或使用 uvicorn：
uvicorn demo_web:app --reload

API 文档：http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from mini_rag import MiniRAG
import uvicorn


# 初始化 FastAPI
app = FastAPI(
    title="最小 RAG Demo API",
    description="一个简单的 RAG 系统 API 演示",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 RAG 实例
rag_instance = None


def get_rag() -> MiniRAG:
    """获取 RAG 实例（单例）"""
    global rag_instance
    
    if rag_instance is None:
        rag_instance = MiniRAG(
            persist_dir="./web_chroma_db",
            collection_name="web_kb",
            chunk_size=300,
            chunk_overlap=60
        )
    
    return rag_instance


# 请求模型
class AddDocumentsRequest(BaseModel):
    """添加文档请求"""
    documents: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None


class QueryRequest(BaseModel):
    """查询请求"""
    question: str
    top_k: int = 3
    use_rerank: bool = True
    rerank_threshold: float = 0.0
    prompt_style: str = "balanced"


# 响应模型
class Document(BaseModel):
    """文档"""
    id: str
    content: str
    vector_similarity: float
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    """查询响应"""
    question: str
    documents: List[Document]
    prompt: str


class StatsResponse(BaseModel):
    """统计响应"""
    total_documents: int
    collection_name: str
    metadata: Dict[str, Any]


# API 端点
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "最小 RAG Demo API",
        "docs": "/docs",
        "endpoints": {
            "POST /documents": "添加文档",
            "POST /query": "查询问题",
            "GET /stats": "查看统计",
            "DELETE /documents": "清空知识库"
        }
    }


@app.post("/documents")
async def add_documents(request: AddDocumentsRequest):
    """
    添加文档到知识库
    
    - **documents**: 文档列表
    - **metadatas**: 元数据列表（可选）
    """
    rag = get_rag()
    
    try:
        rag.add_documents(request.documents, request.metadatas)
        
        return {
            "success": True,
            "message": f"成功添加 {len(request.documents)} 个文档",
            "total_documents": rag.collection.count()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    查询问题
    
    - **question**: 用户问题
    - **top_k**: 返回文档数量
    - **use_rerank**: 是否使用 Rerank
    - **rerank_threshold**: Rerank 分数阈值
    - **prompt_style**: Prompt 风格（strict/balanced/creative）
    """
    rag = get_rag()
    
    try:
        result = rag.query(
            question=request.question,
            top_k=request.top_k,
            use_rerank=request.use_rerank,
            rerank_threshold=request.rerank_threshold,
            prompt_style=request.prompt_style,
            verbose=False
        )
        
        # 转换文档格式
        documents = [
            Document(**doc)
            for doc in result['documents']
        ]
        
        return QueryResponse(
            question=result['question'],
            documents=documents,
            prompt=result['prompt']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """获取知识库统计信息"""
    rag = get_rag()
    stats = rag.get_stats()
    
    return StatsResponse(**stats)


@app.delete("/documents")
async def clear_documents():
    """清空知识库"""
    rag = get_rag()
    
    try:
        rag.clear_collection()
        
        return {
            "success": True,
            "message": "知识库已清空"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    print("\n" + "=" * 60)
    print("🚀 RAG Web API 启动中...")
    print("=" * 60)
    
    # 初始化 RAG
    rag = get_rag()
    
    print("\n" + "=" * 60)
    print("✅ API 启动完成")
    print("=" * 60)
    print("\n📚 API 文档：http://localhost:8000/docs")
    print("🔍 交互式文档：http://localhost:8000/redoc")
    print("=" * 60)


if __name__ == "__main__":
    uvicorn.run(
        "demo_web:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
