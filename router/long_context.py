from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_community.callbacks.manager import get_openai_callback
from pydantic import BaseModel
from typing import List
import time
import json


from ai_model.long_context import stream_long_context, build_full_context

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



router = APIRouter(prefix="/chat/long", tags=["Long Context"])


@router.post("/stream")
def chat_long_stream(request: QuestionRequest):
    def event_generator():
        start = time.time()
        first_token_time = None

        print("Building full context...")
        context = build_full_context()
        if not context.strip():
            yield "I don't know"
            return
        print("Built successfully")

        with get_openai_callback() as cb:
            for token in stream_long_context(context, request.question):
                if first_token_time is None:
                    first_token_time = time.time()
                yield token

        meta = {
            "latency_first_token": round(first_token_time - start, 3),
            "latency_total": round(time.time() - start, 3),
            "tokens": {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens,
                "cost_usd": cb.total_cost,
            },
            "sources": ["FULL_CONTEXT"],
        }

        yield "\n\n[[META]]" + json.dumps(meta)

    return StreamingResponse(event_generator(), media_type="text/plain")
