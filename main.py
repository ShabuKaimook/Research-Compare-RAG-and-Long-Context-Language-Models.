from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pathlib import Path
from vector_db import QdrantStorage
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


from rag.ingest_file import ingest_file
from rag.auto_ingest import auto_ingest


# routers
from router.file import router as file_router
from router.long_context import router as long_context_router
from router.rag import router as rag_router


load_dotenv()


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

# Routers
app.include_router(file_router)
app.include_router(rag_router)
app.include_router(long_context_router)

# Upload directory
UPLOAD_DIR = "./upload"
Path(UPLOAD_DIR).mkdir(exist_ok=True)


@app.on_event("startup")
def startup_event():
    print("🚀 Auto ingest on startup")
    app.state.db = QdrantStorage(collection_name="docs")
    auto_ingest(app.state.db, "./assets")

    # files in upload
    files = [p for p in Path(UPLOAD_DIR).iterdir() if p.is_file()]

    for file in files:
        print(str(file))
        ingest_file(str(file), app.state.db)


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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
