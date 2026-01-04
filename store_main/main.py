import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import (
    RAGChunkAndSrc,
    RAGUpsertResult,
    RAGSearchResult,
    RAGQueryResult,
)

load_dotenv()

# AI_MODEL = "google/gemma-3n-e2b-it:free"
AI_MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"

BASE_URL = "https://openrouter.ai/api/v1"

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
    
)


@inngest_client.create_function(
    fn_id="RAG: Inngest PDF", trigger=inngest.TriggerEvent(event="rag/inngest_pdf")
)

# run the function that in rag_inngest_pdf
async def rag_inngest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        # get source_id from event data or use pdf_path as source_id
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    # upsert the chunks to the vector database (Qdrant)
    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vectors = embed_texts(chunks)
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]
        payloads = [
            {"source": source_id, "text": chunks[i]} for i in range(len(chunks))
        ]

        QdrantStorage().upsert(ids, vectors, payloads)

        return RAGUpsertResult(ingested=len(chunks))

    # load-and-chuck is the name of the step
    # ctx.step.run is the function that run the step like _load in inngest
    chunks_and_src = await ctx.step.run(
        "load-and-chuck", lambda: _load(ctx), output_type=RAGChunkAndSrc
    )
    ingested = await ctx.step.run(
        "embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult
    )

    # take the pandantic model and convert it to a dictionary
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF", trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context) -> RAGSearchResult:
    def _search(question: str, top_k: int = 5):
        # embed the question, store to qdrant and get the vector
        query_vector = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(
            query_vector, top_k
        )  # found will get the context and sources from the search function

        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult,
    )

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    # prompt that will be used to ask LLM
    user_content = (
        "Use this following context to answer the question.\n\n"
        f"{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        model=AI_MODEL, auth_key=os.getenv("OPENROUTER_API_KEY"), base_url=BASE_URL
    )

    response = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_token": 1024,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": "You answer the question using the only provided context",
                },
                {"role": "user", "content": user_content},
            ],
        },
    )

    answer = response["choices"][0]["message"]["content"].strip()

    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts),
    }


app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_inngest_pdf, rag_query_pdf_ai])
