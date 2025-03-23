from langchain_community.vectorstores import FAISS, Qdrant as QdrantVectorStore
from rag.embedding import get_embeddings
from config.settings import FAISS_SAVE_PATH, QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION


def load_faiss_store():
    """加载 FAISS 向量数据库"""
    embeddings = get_embeddings()
    db = FAISS.load_local(FAISS_SAVE_PATH, embeddings, allow_dangerous_deserialization=True)
    print("[DEBUG] FAISS DB loaded:", db)
    return db


def load_qdrant_store():
    """加载 Qdrant 向量数据库"""
    embeddings = get_embeddings()
    db = QdrantVectorStore(
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        collection_name=QDRANT_COLLECTION,
        embeddings=embeddings
    )
    print("[DEBUG] Qdrant DB loaded:", db)
    return db
