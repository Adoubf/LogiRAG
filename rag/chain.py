from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from config.settings import LLM_MODEL
from prompts import qa_prompt


def create_retrieval_chain(retriever):
    """创建问答检索链"""
    print("[DEBUG] 构建 ConversationalRetrievalChain...")
    return ConversationalRetrievalChain.from_llm(
        llm=OllamaLLM(model=LLM_MODEL),
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
