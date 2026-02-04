from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(temperature=0)

prompt = ChatPromptTemplate.from_template(
    """You are ranking document chunks to answer a question.

Question:
{question}

Chunks:
{chunks}

Rules:
- Prefer explanatory text.
- Avoid image captions, diagrams, or figure descriptions.
- Return the FULL TEXT of the best chunks.
- Separate chunks with ---.
"""
)

chain = prompt | llm | StrOutputParser()

def rerank_chunks(question: str, chunks: list[str], top_k: int = 4):
    numbered = "\n\n".join(
        f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks)
    )

    output = chain.invoke(
        {"question": question, "chunks": numbered}
    )

    ranked = [c.strip() for c in output.split("---") if c.strip()]

    # safety: discard very short / caption-like chunks
    return [c for c in ranked if len(c.split()) > 30][:top_k]
