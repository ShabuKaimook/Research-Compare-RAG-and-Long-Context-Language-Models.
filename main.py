from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path
from vector_db import QdrantStorage
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.responses import StreamingResponse
from langchain_community.callbacks.manager import get_openai_callback
import time
import json
import os

from ai_model.rag import ask_rag, stream_rag
from rag.ingest_file import ingest_file
from rag.auto_ingest import auto_ingest
from rag.retriever import advanced_retrieve_context
from ai_model.long_context import stream_long_context, build_full_context
from evaluation.api import router as evaluation_router


load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chat API",
    description="API for uploading documents and asking questions",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = QdrantStorage(collection_name="docs")

def get_db():
    return db

# Upload directory
UPLOAD_DIR = "./upload"
Path(UPLOAD_DIR).mkdir(exist_ok=True)

# files in upload
files = [p for p in Path(UPLOAD_DIR).iterdir() if p.is_file()]

for file in files:
    print(str(file))
    ingest_file(str(file), db)


# Pydantic models
class QuestionRequest(BaseModel):
    question: str


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float  # ✅ MUST be float


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    tokens: TokenUsage
    latency_seconds: float


class UploadResponse(BaseModel):
    filename: str
    size: float
    message: str

app.include_router(evaluation_router)

@app.on_event("startup")
def startup_event():
    print("🚀 Auto ingest on startup")
    auto_ingest(db, "./assets")


# Routes
@app.get("/")
def read_root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to RAG Chat API",
        "endpoints": {
            "upload": "/upload",
            "ask": "/ask",
            "files": "/files",
            "health": "/health",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "database": "connected"}


@app.post("/chat/rag", response_model=ChatResponse)
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


@app.post("/chat/long/stream")
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


@app.get("/files")
def list_files():
    """
    List all uploaded files in the database
    """
    try:
        files = []
        if os.path.exists(UPLOAD_DIR):
            for file in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, file)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path) / 1024
                    files.append({"name": file, "size_kb": round(file_size, 2)})

        return {
            "total_files": len(files),
            "files": sorted(files, key=lambda x: x["name"]),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/files/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT file and ingest it into the vector database.

    Supported formats:
    - PDF (.pdf)
    - Text (.txt)
    """
    try:
        # Validate file type
        allowed_extensions = {".pdf", ".txt"}
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not allowed. Use .pdf or .txt",
            )

        # Save file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)

        # Ingest file to database
        ingest_file(file_path, db)

        file_size = os.path.getsize(file_path) / 1024

        return UploadResponse(
            filename=file.filename,
            size=file_size,
            message="File uploaded and processed successfully",
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/files/{filename}")
def delete_file(filename: str):
    """
    Delete a file from the upload directory
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return {"message": f"File '{filename}' deleted successfully."}
        else:
            raise HTTPException(status_code=404, detail="File not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
