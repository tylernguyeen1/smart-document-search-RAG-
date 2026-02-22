from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.chunk import Chunk, chunk_documents
from src.embed import DEFAULT_EMBED_MODEL, embed_chunks, embed_query
from src.ingest import load_documents


def build_index(
    raw_dir: str | Path,
    index_dir: str | Path,
    chunk_size: int = 1000,
    overlap: int = 200,
    model_name: str = DEFAULT_EMBED_MODEL,
) -> dict[str, Any]:
    raw_path = Path(raw_dir)
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    documents = load_documents(raw_path)
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    embeddings = embed_chunks(
        chunks,
        model_name=model_name,
        cache_path=index_path / "embeddings_cache.json",
    )
    if embeddings.size == 0:
        raise ValueError("No chunks produced from source documents.")

    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    faiss.write_index(faiss_index, str(index_path / "faiss.index"))
    metadata = {
        "model_name": model_name,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "count": len(chunks),
        "chunks": [asdict(c) for c in chunks],
    }
    (index_path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return metadata


def load_index(index_dir: str | Path) -> tuple[faiss.Index, dict[str, Any]]:
    index_path = Path(index_dir)
    faiss_path = index_path / "faiss.index"
    meta_path = index_path / "metadata.json"
    if not faiss_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing index files in {index_path}. Run scripts/build_index.py first."
        )

    faiss_index = faiss.read_index(str(faiss_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return faiss_index, metadata


def search(
    query: str,
    index_dir: str | Path,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    faiss_index, metadata = load_index(index_dir)
    model_name = metadata.get("model_name", DEFAULT_EMBED_MODEL)
    q = embed_query(query, model_name=model_name)
    scores, indices = faiss_index.search(q, top_k)

    chunks = metadata["chunks"]
    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
        if idx < 0:
            continue
        chunk: dict[str, Any] = chunks[idx]
        results.append(
            {
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "file_name": chunk["file_name"],
                "source_path": chunk["source_path"],
                "text": chunk["text"],
            }
        )
    return results


def draft_answer(query: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    if not retrieved_chunks:
        return "I could not find supporting evidence in the indexed documents."
    best = retrieved_chunks[0]
    return (
        f"Best match for '{query}' is from {best['file_name']} ({best['chunk_id']}): "
        f"{best['text'][:280]}..."
    )


def _extract_summary_sentences(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    selected: list[str] = []
    used = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        add = len(sentence) + (1 if selected else 0)
        if used + add > max_chars:
            break
        selected.append(sentence)
        used += add
    return selected


def summarize_results(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    max_chars: int = 500,
    answer_format: str = "paragraph",
) -> str:
    if not retrieved_chunks:
        return "I could not find supporting evidence in the indexed documents."

    combined = " ".join(chunk["text"] for chunk in retrieved_chunks[:3]).strip()
    if not combined:
        return "I could not find supporting evidence in the indexed documents."

    selected = _extract_summary_sentences(combined, max_chars=max_chars)

    if selected:
        snippet = " ".join(selected)
    else:
        snippet = combined[:max_chars].strip()
        if len(combined) > max_chars:
            snippet += "..."

    if answer_format == "bullets":
        bullet_sentences = _extract_summary_sentences(combined, max_chars=max_chars)
        if not bullet_sentences:
            bullet_sentences = [snippet]
        bullets = "\n".join(f"- {sentence}" for sentence in bullet_sentences)
        return f"Summary for '{query}':\n{bullets}"

    return f"Summary for '{query}': {snippet}"
