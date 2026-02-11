from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_community.callbacks.manager import get_openai_callback
import time
import json

from ai_model.rag import ask_rag, stream_rag
from rag.retriever import advanced_retrieve_context
from pydantic import BaseModel
from typing import List

# Pydantic models
class QuestionRequest(BaseModel):
    question: str


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    tokens: TokenUsage
    latency_seconds: float


class UploadResponse(BaseModel):
    filename: str
    size: float
    message: str



router = APIRouter(prefix="/chat/rag", tags=["Rag"])


@router.post("/", response_model=ChatResponse)
def chat_rag(request: QuestionRequest):
    """
    Ask a question about the uploaded documents.

    Uses advanced retrieval with query rewriting, deduplication, and re-ranking.
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        result = ask_rag(request.question, db)

        print("Answer:", result["answer"])
        print("Sources:", result["sources"])
        print("Tokens:", result["tokens"])
        print("Latency:", result["latency_seconds"])

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            tokens=result["tokens"],
            latency_seconds=result["latency_seconds"],
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/rag/stream")
def chat_rag_stream(request: QuestionRequest):
    def event_generator():
        start = time.time()
        first_token_time = None

        context, sources = advanced_retrieve_context(request.question, db)

        if not context:
            yield "I don't know"
            return

        with get_openai_callback() as cb:
            for token in stream_rag(context, request.question):
                if first_token_time is None:
                    first_token_time = time.time()
                yield token

        # 🔴 META MUST BE SENT HERE
        meta = {
            "latency_first_token": round(first_token_time - start, 3)
            if first_token_time
            else None,
            "latency_total": round(time.time() - start, 3),
            "tokens": {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens,
                "cost_usd": cb.total_cost,
            },
            "sources": sources,
        }

        yield "\n\n[[META]]" + json.dumps(meta)

        print("META:", meta)

    return StreamingResponse(event_generator(), media_type="text/plain")