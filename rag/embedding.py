from langchain_ollama import OllamaEmbeddings
from config.settings import EMBEDDING_MODEL


def get_embeddings():
    """获取Embedding模型实例"""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)
