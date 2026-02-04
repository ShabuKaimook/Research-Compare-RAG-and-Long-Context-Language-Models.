from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

from vector_db import QdrantStorage
from rag.retriever import advanced_retrieve_context
from rag.ingest_file import ingest_file

load_dotenv()

llm = ChatOpenAI(
    model_name=os.getenv("OPENAI_MODEL"),
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY"),
)

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant.
You may translate the question if needed.
Answer the question using ONLY the context below.
If the context does not directly answer the question,
you may summarize or explain based on the information available.
If the context is completely unrelated, say "I don't know".

Context:
{context}

Question:
{question}
"""
)

chain = prompt | llm | StrOutputParser()

db = QdrantStorage(collection_name="docs")


def ask_ai(question: str):
    context, sources = advanced_retrieve_context(question, db)

    print("Retrieved context:", context[:100])

    if not context:
        return {
            "answer": "I don't know",
            "sources": [],
        }

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    print("Answer in chat.py:", answer)
    

    return {
        "answer": answer,
        "sources": sources,
    }


# if __name__ == "__main__":
#     ingest_file("./upload/trigon.pdf", db)
#     ingest_file("./upload/Nvidia.txt", db)

    # result = ask_ai("สรุป trigon ให้ฟังหน่อย")
    # print("Answer:")
    # print(result["answer"])
    # print("Sources:", result["sources"])
