from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from functools import lru_cache

llm = ChatOpenAI(temperature=0)

prompt = ChatPromptTemplate.from_template(
    """You are a search query rewriting assistant.

Given the user question, generate 2 fixed grammatically correct search queries in the most appropriate languages to retrieve relevant information.

User question:
{question}

Return each query on a new line.
"""
)

chain = prompt | llm | StrOutputParser()


@lru_cache(maxsize=128)
def rewrite_query(question: str):
    output = chain.invoke({"question": question})
    return [q.strip() for q in output.splitlines() if q.strip()]
