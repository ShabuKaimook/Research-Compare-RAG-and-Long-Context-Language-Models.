from fastapi import APIRouter, HTTPException, Request, Depends, UploadFile, File
import os
from pathlib import Path
from pydantic import BaseModel
from typing import List
from rag.ingest_file import ingest_file

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



router = APIRouter(prefix="/files", tags=["Files"])
UPLOAD_DIR = "./upload"
Path(UPLOAD_DIR).mkdir(exist_ok=True)

def get_db(request: Request):
    return request.app.state.db

@router.get("/")
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


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), db = Depends(get_db)):
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


@router.delete("/files/{filename}")
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
