"""
集中配置项管理
"""

# 数据相关配置
DATA_DIR = "./dataset"
PROCESSED_RECORD = "./logs/processed_files.json"
FAISS_SAVE_PATH = "./faiss/multi_format_store"

# Qdrant配置
QDRANT_COLLECTION = "multi_format_vectors"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# 模型配置
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama2-chinese:7b"

# 文本分割配置
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# 并发和缓存配置
MAX_WORKERS = 4  # 并发工作进程数量
EMBEDDING_CACHE_DIR = "./cache/embeddings"  # 嵌入向量缓存目录
PROCESSING_STATE_FILE = "./logs/processing_state.json"  # 断点续传状态文件
BATCH_SIZE = 100  # 批处理大小，用于向量操作
