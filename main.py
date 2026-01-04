import logging
import os
import uuid
from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from fastapi.responses import StreamingResponse
import json
import time

import inngest
import inngest.fast_api

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import (
    RAGChunkAndSrc,
    RAGUpsertResult,
)

# Config

load_dotenv()

AI_MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"
BASE_URL = "https://openrouter.ai/api/v1"

# OpenAI client (OpenRouter)
openai_client = OpenAI(
    base_url=BASE_URL,
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Inngest client (for ingestion only)
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


# Inngest Function: PDF Ingestion (BACKGROUND)
@inngest_client.create_function(
    fn_id="RAG: Inngest PDF",
    trigger=inngest.TriggerEvent(event="rag/inngest_pdf"),
)

# run the function that in rag_inngest_pdf
async def rag_inngest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        # get source_id from event data or use pdf_path as source_id
        pdf_path = ctx.event.data["pdf_path"]
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
        "load-and-chunk",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc,
    )

    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult,
    )

    # take the pandantic model and convert it to a dictionary
    return ingested.model_dump()


# FastAPI App
app = FastAPI(title="RAG API")


# Sync RAG Chat Endpoint (FAST)
class RAGQueryRequest(BaseModel):
    question: str
    top_k: int = 3


@app.post("/rag/query")
async def rag_query_sync(payload: RAGQueryRequest):
    # embed the question, store to qdrant and get the vector
    query_vector = embed_texts([payload.question])[0]

    # found will get the context and sources from the search function
    store = QdrantStorage()
    found = store.search(query_vector, payload.top_k)

    contexts = found.get("contexts", [])
    if not contexts:
        return {
            "answer": "No documents found. Please ingest a PDF first.",
            "sources": [],
            "num_contexts": 0,
        }

    # prompt the LLM with the context and question
    context_block = "\n\n".join(f"- {c}" for c in contexts)
    user_content = (
        "Use the context below to answer the question.\n\n"
        f"{context_block}\n\n"
        f"Question: {payload.question}"
    )

    # call the LLM
    completion = openai_client.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {"role": "system", "content": "Answer only from context"},
            {"role": "user", "content": user_content},
        ],
        max_tokens=1024,
        temperature=0.2,
    )

    answer = completion.choices[0].message.content.strip()

    return {
        "answer": answer,
        "sources": found.get("sources", []),
        "num_contexts": len(contexts),
    }


@app.post("/rag/query/stream")
async def rag_query_stream(payload: RAGQueryRequest):
    t_start = time.perf_counter()

	# embed the question, store to qdrant and get the vector
    query_vector = embed_texts([payload.question])[0]
    store = QdrantStorage()

	# found will get the context and sources from the search function
    found = store.search(query_vector, payload.top_k)

    context_block = "\n\n".join(f"- {c}" for c in found["contexts"])
    prompt = (
        "Use the context below to answer the question.\n"
        "If math is used, output LaTeX using $$ ... $$ only.\n\n"
        f"{context_block}\n\n"
        f"Question: {payload.question}"
		"Answer concisely using the context above."
    )

    t_first_token = None

    def generator():
        nonlocal t_first_token

        stream = openai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": "Answer only from the given context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                    backend_ttfb = t_first_token - t_start
                yield f"<<TTFB:{backend_ttfb:.3f}>>"

            yield delta.content

        # หลัง stream จบ ค่อยวัด total
        t_end = time.perf_counter()
        backend_total = t_end - t_start
        yield f"<<TOTAL:{backend_total:.3f}>>"

    def headers():
        t_end = time.perf_counter()
        return {
            "X-Sources": json.dumps(found["sources"]),
            "X-Backend-Total-Time": f"{t_end - t_start:.3f}",
            "X-Backend-TTFB": (
                f"{t_first_token - t_start:.3f}" if t_first_token else "N/A"
            ),
        }

    return StreamingResponse(
        generator(),
        media_type="text/plain",
        headers=headers(),
    )


# Health check
@app.get("/")
def health():
    return {"status": "ok"}


# Serve Inngest + FastAPI together
inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[rag_inngest_pdf],
)
