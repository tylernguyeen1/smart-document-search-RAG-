from __future__ import annotations

from dataclasses import dataclass

from src.ingest import Document


@dataclass
class Chunk:
    chunk_id: str
    file_name: str
    source_path: str
    text: str


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    words = cleaned.split(" ")
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    total_words = len(words)
    while start < total_words:
        current_words: list[str] = []
        current_len = 0
        idx = start

        # Build a chunk up to chunk_size while preserving whole words.
        while idx < total_words:
            word = words[idx]
            added = len(word) if not current_words else len(word) + 1
            if current_words and (current_len + added > chunk_size):
                break
            current_words.append(word)
            current_len += added
            idx += 1

        if not current_words:
            break

        chunks.append(" ".join(current_words))

        if idx >= total_words:
            break

        # Move start backward to preserve approximately `overlap` characters.
        back_chars = 0
        new_start = idx
        while new_start > start and back_chars < overlap:
            new_start -= 1
            back_chars += len(words[new_start]) + 1

        if new_start <= start:
            start = idx
        else:
            start = new_start
    return chunks


def chunk_documents(
    documents: list[Document], chunk_size: int = 1000, overlap: int = 200
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in documents:
        parts = split_text(doc.text, chunk_size=chunk_size, overlap=overlap)
        for idx, part in enumerate(parts):
            out.append(
                Chunk(
                    chunk_id=f"{doc.file_name}:{idx}",
                    file_name=doc.file_name,
                    source_path=doc.source_path,
                    text=part,
                )
            )
    return out
