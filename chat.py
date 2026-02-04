from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback
from dotenv import load_dotenv
import os
import time

from vector_db import QdrantStorage
from rag.retriever import advanced_retrieve_context
# from rag.ingest_file import ingest_file

load_dotenv()

llm = ChatOpenAI(
    model_name=os.getenv("OPENAI_MODEL"),
    temperature=1,
    streaming=True,
    api_key=os.getenv("OPENAI_API_KEY"),
)

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant.

You may translate the question if needed.
Answer the question using ONLY the context below.

If the context does not directly answer the question,
you may summarize or explain based on the information available.

If the question ask to translate the context to another language, you may translate the context to the language asked.

If the question ask in the language that different from the context, when you answer, you should translate the answer to the language asked.

If the context is completely unrelated, say "The question is unrelated to the context".

Context:
{context}

Question:
{question}
"""
)

chain = prompt | llm | StrOutputParser()

db = QdrantStorage(collection_name="docs")

def ask_ai(question: str):
    start_time = time.time()
    context, sources = advanced_retrieve_context(question, db)

    print("Retrieved context:", context[:100])

    if not context:
        return {
            "answer": "I don't know",
            "sources": [],
            "tokens": 0,
            "latency": 0,
        }

    with get_openai_callback() as cb:
        answer = chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

    print("Answer in chat.py:", answer)
    
    latency = time.time() - start_time
    return {
        "answer": answer,
        "sources": sources,
        "tokens": {
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_tokens": cb.total_tokens,
            "cost_usd": cb.total_cost,
        },
        "latency_seconds": round(latency, 3),
    }

def stream_answer(context: str, question: str):
    messages = [
        {"role": "system", "content": "Answer using the context only"},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
    ]

    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content



# if __name__ == "__main__":
#     ingest_file("./upload/trigon.pdf", db)
#     ingest_file("./upload/Nvidia.txt", db)

    # result = ask_ai("สรุป trigon ให้ฟังหน่อย")
    # print("Answer:")
    # print(result["answer"])
    # print("Sources:", result["sources"])
