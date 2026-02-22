from __future__ import annotations

from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.index import build_index, search, summarize_results

app = FastAPI(title="Smart Document Search API", version="0.1.0")
DATA_RAW_DIR = Path("data/raw")
DATA_INDEX_DIR = Path("data/index")

default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
extra_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
allowed_origins = default_origins + extra_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BuildRequest(BaseModel):
    raw_dir: str = "data/raw"
    index_dir: str = "data/index"
    chunk_size: int = 1000
    overlap: int = 200


class QueryRequest(BaseModel):
    query: str
    index_dir: str = str(DATA_INDEX_DIR)
    top_k: int = 5
    answer_format: str = "paragraph"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/build-index")
def build_index_endpoint(payload: BuildRequest) -> dict:
    try:
        meta = build_index(
            raw_dir=payload.raw_dir,
            index_dir=payload.index_dir,
            chunk_size=payload.chunk_size,
            overlap=payload.overlap,
        )
        return {"message": "index built", "metadata": meta}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/upload-and-index")
async def upload_and_index(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, .md are supported.")

    try:
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        target = DATA_RAW_DIR / file.filename
        content = await file.read()
        target.write_bytes(content)

        meta = build_index(raw_dir=DATA_RAW_DIR, index_dir=DATA_INDEX_DIR)
        return {
            "message": "file uploaded and index rebuilt",
            "file_name": file.filename,
            "metadata": {
                "count": meta["count"],
                "chunk_size": meta["chunk_size"],
                "overlap": meta["overlap"],
                "model_name": meta["model_name"],
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask")
def ask_endpoint(payload: QueryRequest) -> dict:
    try:
        answer_format = payload.answer_format.lower().strip()
        if answer_format not in {"paragraph", "bullets"}:
            raise ValueError("answer_format must be 'paragraph' or 'bullets'.")
        results = search(query=payload.query, index_dir=payload.index_dir, top_k=payload.top_k)
        return {
            "summary": summarize_results(
                payload.query, results, answer_format=answer_format
            ),
            "results": results,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/query")
def query_endpoint(payload: QueryRequest) -> dict:
    try:
        answer_format = payload.answer_format.lower().strip()
        if answer_format not in {"paragraph", "bullets"}:
            raise ValueError("answer_format must be 'paragraph' or 'bullets'.")
        results = search(query=payload.query, index_dir=payload.index_dir, top_k=payload.top_k)
        return {
            "answer": summarize_results(payload.query, results, answer_format=answer_format),
            "results": results,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
