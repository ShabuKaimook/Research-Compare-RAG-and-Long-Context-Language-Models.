from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding_model = OpenAIEmbeddings(
    # base_url="https://openrouter.ai/api/v1",
    model=os.getenv("OPENAI_EMBEDDING_MODEL"),
    dimensions=1536,
    api_key=os.getenv("OPENAI_API_KEY"),
)


def embed_docs(chunks):
    texts = [c.page_content for c in chunks]
    return embedding_model.embed_documents(texts)


def embed_query(query: str):
    return embedding_model.embed_query(query)
