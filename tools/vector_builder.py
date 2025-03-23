"""
向量数据库构建工具
"""
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import json
import hashlib
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Any, Callable

# 导入LangChain相关模块
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# 使用新的推荐的QdrantVectorStore类
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)
from langchain.embeddings.base import Embeddings

# 导入第三方库
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# 导入自定义模块
from config.settings import (
    DATA_DIR, PROCESSED_RECORD, FAISS_SAVE_PATH,
    QDRANT_COLLECTION, QDRANT_HOST, QDRANT_PORT,
    CHUNK_SIZE, CHUNK_OVERLAP, MAX_WORKERS,
    EMBEDDING_CACHE_DIR, PROCESSING_STATE_FILE
)
from rag.embedding import get_embeddings
from utils.logging_utils import setup_logger
from utils.cache_utils import EmbeddingCache, ProcessingStateManager

# 初始化
logger = setup_logger(log_file="logs/vector_build.log")
os.makedirs("logs", exist_ok=True)

# 创建文本分割器和嵌入模型
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
embeddings = get_embeddings()

# 初始化缓存和状态管理器
embedding_cache = EmbeddingCache(EMBEDDING_CACHE_DIR)
state_manager = ProcessingStateManager(PROCESSING_STATE_FILE)


# 初始化Qdrant
def init_qdrant():
    # 获取 embedding 维度（动态）
    example_vector = embeddings.embed_query("dummy")
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    if not qdrant_client.collection_exists(QDRANT_COLLECTION):
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=len(example_vector), distance=Distance.COSINE)
        )
    else:
        logger.info(f"[INFO] Qdrant collection '{QDRANT_COLLECTION}' 已存在，跳过创建。")
    return qdrant_client


def get_file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def load_processed_files():
    if os.path.exists(PROCESSED_RECORD):
        if os.path.getsize(PROCESSED_RECORD) == 0:
            return {}
        try:
            with open(PROCESSED_RECORD, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"[WARN] 记录文件损坏，已忽略：{PROCESSED_RECORD}")
            return {}
    return {}


def save_processed_files(record):
    with open(PROCESSED_RECORD, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


class CachedEmbeddings(Embeddings):
    """带缓存的嵌入模型包装器，兼容 LangChain Embeddings 接口"""

    def __init__(self, embedding_model, cache):
        self.embedding_model = embedding_model
        self.cache = cache
        self.cache_hits = 0
        self.cache_misses = 0

    def embed_query(self, text: str) -> List[float]:
        cached = self.cache.get(text)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self.cache_misses += 1
        embedding = self.embedding_model.embed_query(text)
        self.cache.set(text, embedding)
        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def get_stats(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total) * 100 if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total": total,
            "hit_rate": hit_rate
        }

    def __call__(self, text):
        if isinstance(text, list):
            return self.embed_documents(text)
        return self.embed_query(text)


def process_file(path):
    filename = os.path.basename(path)
    try:
        if filename.endswith(".pdf"):
            docs = PyMuPDFLoader(path).load()
        elif filename.endswith(".docx"):
            docs = UnstructuredWordDocumentLoader(path).load()
        elif filename.endswith(".xlsx"):
            docs = UnstructuredExcelLoader(path).load()
        elif filename.endswith(".csv"):
            try:
                df = pd.read_csv(path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="gbk")
            docs = [Document(page_content=row.to_string(index=False), metadata={"source": filename}) for _, row in
                    df.iterrows()]
        elif filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            docs = [Document(page_content=text, metadata={"source": filename})]
        else:
            logger.warning(f"[SKIP] Unsupported format: {filename}")
            return []
        logger.info(f"[LOAD] {filename}: {len(docs)} docs")
        return docs
    except Exception as e:
        logger.error(f"[ERROR] Failed to process {filename}: {e}")
        return []


def build_vector_stores(documents, append_mode=False):
    """
    构建向量存储

    Args:
        documents: 文档列表
        append_mode: 是否追加到现有集合
    """
    if not documents:
        logger.info("没有文档需要处理")
        return

    chunks = splitter.split_documents(documents)
    logger.info(f"[INFO] 文本分割后共 {len(chunks)} 个块")

    # 使用缓存包装嵌入模型
    cached_embeddings = CachedEmbeddings(embeddings, embedding_cache)

    # FAISS
    if append_mode and os.path.exists(FAISS_SAVE_PATH):
        logger.info(f"[INFO] 追加到现有FAISS索引: {FAISS_SAVE_PATH}")
        try:
            faiss_store = FAISS.load_local(
                FAISS_SAVE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            faiss_store.add_documents(chunks)
        except Exception as e:
            logger.error(f"[ERROR] 追加到FAISS失败，创建新索引: {e}")
            faiss_store = FAISS.from_documents(chunks, embeddings)
    else:
        faiss_store = FAISS.from_documents(chunks, embeddings)

    faiss_store.save_local(FAISS_SAVE_PATH)
    logger.info(f"[SAVE] FAISS索引已保存，共 {faiss_store.index.ntotal} 个向量")

    # Qdrant - 使用新的推荐API
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if append_mode and client.collection_exists(QDRANT_COLLECTION):
        logger.info(f"[INFO] 追加到现有Qdrant集合: {QDRANT_COLLECTION}")
        
        # 使用新的QdrantVectorStore类，正确设置embeddings参数
        qdrant_store = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embeddings=cached_embeddings  # 使用embeddings参数而非embedding_function
        )
        qdrant_store.add_documents(chunks)
        logger.info(f"[INFO] 已向Qdrant集合追加 {len(chunks)} 个文档")
    else:
        if client.collection_exists(QDRANT_COLLECTION):
            logger.info(f"[INFO] 删除现有Qdrant集合: {QDRANT_COLLECTION}")
            client.delete_collection(QDRANT_COLLECTION)

        # 使用新的QdrantVectorStore类的from_documents方法
        qdrant_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=cached_embeddings,  # 注意这里使用embedding参数名
            url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
            collection_name=QDRANT_COLLECTION
        )

    # 缓存统计
    cache_stats = cached_embeddings.get_stats()
    logger.info(
        f"[INFO] 嵌入缓存统计: 命中={cache_stats['hits']}, 未命中={cache_stats['misses']}, 命中率={cache_stats['hit_rate']:.2f}%"
    )
    logger.info(f"[SAVE] Qdrant向量已保存到集合: {QDRANT_COLLECTION}")


def process_files_with_state(file_paths, append_mode=False, resume=False):
    """
    带断点续传的文件处理
    
    Args:
        file_paths: 文件路径列表
        append_mode: 是否追加到现有集合
        resume: 是否从上次中断点继续
    """
    all_docs = []
    processed_files = {}
    current_state = None

    # 如果需要恢复，尝试加载之前的状态
    if resume:
        current_state = state_manager.load_state()
        if current_state:
            logger.info(f"[INFO] 从断点继续，已处理 {len(current_state['processed'])} 个文件")
            processed_files = current_state["processed"]

            # 过滤掉已处理的文件
            file_paths = [(path, f_hash) for path, f_hash in file_paths
                          if os.path.basename(path) not in processed_files]

    if not file_paths:
        logger.info("[INFO] 没有需要处理的文件")
        return

    logger.info(f"[INFO] 开始处理 {len(file_paths)} 个文件，使用 {MAX_WORKERS} 个工作进程")

    # 并发处理文件，但控制并发数量
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file, path): (path, f_hash) for path, f_hash in file_paths}

        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理文件"):
                docs = future.result()
                path, f_hash = futures[future]
                filename = os.path.basename(path)

                if docs:
                    all_docs.extend(docs)
                    processed_files[filename] = f_hash

                    # 定期保存状态
                    state_manager.save_state({
                        "processed": processed_files,
                        "timestamp": pd.Timestamp.now().isoformat()
                    })

                    logger.info(f"[INFO] 已处理: {filename}, 提取了 {len(docs)} 个文档, 总计 {len(all_docs)} 个文档")
                else:
                    logger.warning(f"[WARN] 处理失败或无内容: {filename}")

        except KeyboardInterrupt:
            logger.warning("[WARN] 用户中断，保存当前状态...")
            # 保存当前状态
            state_manager.save_state({
                "processed": processed_files,
                "timestamp": pd.Timestamp.now().isoformat(),
                "interrupted": True
            })
            logger.info("[INFO] 状态已保存，可以使用 --resume 选项稍后继续")
            return

    # 如果有文档，构建向量存储
    if all_docs:
        build_vector_stores(all_docs, append_mode=append_mode)

        # 处理完成后，更新记录并清除状态
        save_processed_files({**load_processed_files(), **processed_files})
        state_manager.clear_state()

        logger.info(f"[INFO] 完成。共处理 {len(processed_files)} 个文件，嵌入 {len(all_docs)} 个文档")
    else:
        logger.warning("[WARN] 没有有效的文档被处理")


def main(append_mode=False, resume=False, clear_cache=False):
    """
    主函数
    
    Args:
        append_mode: 是否追加到现有集合
        resume: 是否从断点继续
        clear_cache: 是否清除嵌入缓存
    """
    # 清除缓存（如果需要）
    if clear_cache:
        logger.info("[INFO] 清除嵌入缓存...")
        embedding_cache.clear()

    # 确保Qdrant已初始化
    init_qdrant()

    # 加载已处理文件记录
    processed = load_processed_files()
    file_paths = []

    # 增量判断
    for f in os.listdir(DATA_DIR):
        full_path = os.path.join(DATA_DIR, f)
        if not os.path.isfile(full_path):
            continue

        f_hash = get_file_hash(full_path)

        # 如果是追加模式，或者文件已更改，则处理
        if append_mode or processed.get(f) != f_hash:
            file_paths.append((full_path, f_hash))
        else:
            logger.info(f"[SKIP] 未更改: {f}")

    if not file_paths:
        logger.info("[INFO] 没有新的或更新的文件需要处理")
        return

    # 使用带断点续传的处理函数
    process_files_with_state(file_paths, append_mode=append_mode, resume=resume)


if __name__ == "__main__":
    # 支持直接运行此脚本
    import argparse

    parser = argparse.ArgumentParser(description="向量数据库构建工具")
    parser.add_argument('--append', action='store_true', help="追加到现有集合，而非重建")
    parser.add_argument('--resume', action='store_true', help="从上次中断点继续")
    parser.add_argument('--clear-cache', action='store_true', help="清除嵌入缓存")

    args = parser.parse_args()

    main(append_mode=args.append, resume=args.resume, clear_cache=args.clear_cache)
