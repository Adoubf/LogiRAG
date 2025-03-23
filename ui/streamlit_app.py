import streamlit as st
from data.vector_store import load_faiss_store
from rag.chain import create_retrieval_chain


def setup_ui():
    """设置Streamlit界面"""
    st.set_page_config(page_title="物流行业信息咨询系统", layout="wide")
    st.title("🚚 物流行业信息咨询系统")


def run_app():
    """运行Streamlit应用"""
    setup_ui()

    # 加载向量库
    db = load_faiss_store()
    # 全局上下文历史
    chat_history = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt_text := st.chat_input("请输入你的问题:"):
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # 显式验证向量库是否命中
            matched_docs = db.similarity_search(prompt_text, k=2)
            print("[DEBUG] 相似文档匹配结果：")
            for i, doc in enumerate(matched_docs):
                print(f"Doc {i + 1}: {doc.page_content}")

            chain = create_retrieval_chain(db.as_retriever())
            result = chain.invoke({"question": prompt_text, "chat_history": chat_history})

            answer = result["answer"]
            chat_history.append((prompt_text, answer))

            for chunk in answer.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            print("[DEBUG] 模型回答：", answer)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
