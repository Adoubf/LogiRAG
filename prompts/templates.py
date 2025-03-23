from langchain_core.prompts import PromptTemplate

# 自定义提示模板（强制防止模型发挥）
PROMPT_TEMPLATE = """
你必须只根据以下已知信息来回答用户问题，不允许添加编造内容，不允许建议用户去搜索或查询其它来源！

已知内容:
{context}

问题:
{question}
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE,
)
