"""
向量数据管理工具
提供向量删除、增量追加等功能
"""
import os
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchRequest
from langchain_community.vectorstores import FAISS, Qdrant as QdrantVectorStore

from config.settings import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, 
    FAISS_SAVE_PATH, BATCH_SIZE
)
from rag.embedding import get_embeddings
from utils.logging_utils import setup_logger

# 初始化日志记录器
logger = setup_logger(log_file="logs/vector_manager.log")

def get_qdrant_client() -> QdrantClient:
    """获取Qdrant客户端连接"""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def delete_by_source(source_names: List[str]) -> int:
    """
    根据源文件名删除向量
    
    Args:
        source_names: 源文件名列表
        
    Returns:
        删除的向量数量
    """
    client = get_qdrant_client()
    
    # 检查集合是否存在
    if not client.collection_exists(QDRANT_COLLECTION):
        logger.warning(f"集合不存在: {QDRANT_COLLECTION}")
        return 0
    
    # 构建过滤条件
    filter_condition = Filter(
        should=[
            FieldCondition(
                key="metadata.source",
                match=MatchValue(value=source)
            )
            for source in source_names
        ]
    )
    
    try:
        # 获取向量维度
        collection_info = client.get_collection(QDRANT_COLLECTION)
        vector_size = collection_info.config.params.vectors.size
        dummy_vector = [0.0] * vector_size  # 创建全零向量
        
        logger.info(f"搜索符合条件的向量: {source_names}")
        
        # 使用搜索操作查找符合条件的向量
        search_result = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=dummy_vector,  # 使用零向量作为查询向量
            query_filter=filter_condition,
            limit=10000,  # 获取足够多的匹配点
            with_payload=False,
            with_vectors=False
        )
        
        # 提取ID
        point_ids = [point.id for point in search_result]
        
        if not point_ids:
            logger.info(f"没有找到匹配的向量: {source_names}")
            return 0
        
        logger.info(f"找到 {len(point_ids)} 个匹配的向量")
        
        # 分批删除点
        deleted_count = 0
        for i in range(0, len(point_ids), BATCH_SIZE):
            batch = point_ids[i:i+BATCH_SIZE]
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=batch
            )
            deleted_count += len(batch)
            logger.info(f"已删除 {deleted_count}/{len(point_ids)} 个向量")
        
        # 更新FAISS索引
        logger.info("正在从Qdrant导出数据以更新FAISS...")
        embeddings = get_embeddings()
        
        # 获取集合中的点数量
        count_result = client.count(QDRANT_COLLECTION)
        if count_result.count > 0:
            # 使用简单的方法重建FAISS索引
            try:
                # 从剩余的数据创建QdrantVectorStore
                qdrant_store = QdrantVectorStore(
                    client=client,
                    collection_name=QDRANT_COLLECTION,
                    embeddings=embeddings
                )
                
                # 搜索一些点以获取文档
                search_results = client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=dummy_vector,
                    limit=10000,
                    with_payload=True
                )
                
                if search_results:
                    # 从搜索结果中提取文档
                    from langchain_core.documents import Document
                    docs = []
                    for point in search_results:
                        if point.payload and "page_content" in point.payload:
                            metadata = point.payload.get("metadata", {})
                            docs.append(Document(
                                page_content=point.payload["page_content"],
                                metadata=metadata
                            ))
                    
                    # 创建新的FAISS索引
                    if docs:
                        faiss_store = FAISS.from_documents(docs, embeddings)
                        faiss_store.save_local(FAISS_SAVE_PATH)
                        logger.info(f"FAISS索引已更新，共 {faiss_store.index.ntotal} 个向量")
                    else:
                        logger.warning("未找到有效文档，FAISS索引未更新")
                else:
                    logger.warning("搜索结果为空，FAISS索引未更新")
            except Exception as e:
                logger.error(f"更新FAISS索引时出错: {e}")
        else:
            # 如果集合为空，清空FAISS索引
            logger.info("Qdrant集合为空，创建空的FAISS索引")
            from langchain_core.documents import Document
            empty_docs = [Document(page_content="placeholder", metadata={})]
            faiss_store = FAISS.from_documents(empty_docs, embeddings)
            faiss_store.save_local(FAISS_SAVE_PATH)
            logger.info("已创建空的FAISS索引")
        
        return deleted_count
    
    except Exception as e:
        logger.error(f"删除向量时发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0

def get_collection_stats() -> Dict[str, Any]:
    """
    获取向量集合统计信息
    
    Returns:
        集合统计信息字典
    """
    client = get_qdrant_client()
    
    if not client.collection_exists(QDRANT_COLLECTION):
        return {"exists": False, "count": 0, "vectors": {}}
    
    # 获取集合信息
    collection_info = client.get_collection(QDRANT_COLLECTION)
    
    # 获取集合中的点数量
    count_result = client.count(QDRANT_COLLECTION)
    
    stats = {
        "exists": True,
        "count": count_result.count,
        "vectors": {
            "size": collection_info.config.params.vectors.size,
            "distance": collection_info.config.params.vectors.distance
        }
    }
    
    # 加载FAISS信息
    if os.path.exists(FAISS_SAVE_PATH):
        embeddings = get_embeddings()
        try:
            faiss_store = FAISS.load_local(FAISS_SAVE_PATH, embeddings)
            stats["faiss"] = {
                "exists": True,
                "count": faiss_store.index.ntotal
            }
        except Exception as e:
            stats["faiss"] = {
                "exists": True,
                "error": str(e)
            }
    else:
        stats["faiss"] = {
            "exists": False
        }
    
    return stats

def list_source_files() -> List[str]:
    """
    列出向量集合中的所有源文件
    
    Returns:
        源文件名列表
    """
    client = get_qdrant_client()
    
    if not client.collection_exists(QDRANT_COLLECTION):
        return []
    
    # 获取所有点的元数据 - 修复API兼容性问题
    # 移除不支持的 payload_selector 参数，改为获取完整的payload
    search_result = client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )
    
    # 提取唯一的源文件名
    sources = set()
    for point in search_result[0]:
        if point.payload and "metadata" in point.payload and "source" in point.payload["metadata"]:
            sources.add(point.payload["metadata"]["source"])
    
    return sorted(list(sources))
