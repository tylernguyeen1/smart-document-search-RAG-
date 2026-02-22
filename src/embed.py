from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunk import Chunk

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache(cache_path: Path) -> dict[str, list[float]]:
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _save_cache(cache_path: Path, cache: dict[str, list[float]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")


def embed_chunks(
    chunks: list[Chunk],
    model_name: str = DEFAULT_EMBED_MODEL,
    cache_path: str | Path | None = None,
) -> np.ndarray:
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)

    model = SentenceTransformer(model_name)
    texts = [c.text for c in chunks]
    cache_file = Path(cache_path) if cache_path else None
    cache: dict[str, list[float]] = _load_cache(cache_file) if cache_file else {}

    vectors = []
    missing_texts = []
    missing_keys = []
    for text in texts:
        key = _hash_text(text)
        if key in cache:
            vectors.append(cache[key])
        else:
            vectors.append(None)
            missing_texts.append(text)
            missing_keys.append(key)

    if missing_texts:
        new_vectors = model.encode(missing_texts, normalize_embeddings=True)
        for key, vec in zip(missing_keys, new_vectors):
            cache[key] = vec.tolist()

    final = []
    for idx, text in enumerate(texts):
        if vectors[idx] is None:
            vectors[idx] = cache[_hash_text(text)]
        final.append(vectors[idx])

    if cache_file:
        _save_cache(cache_file, cache)

    return np.asarray(final, dtype=np.float32)


def embed_query(query: str, model_name: str = DEFAULT_EMBED_MODEL) -> np.ndarray:
    model = SentenceTransformer(model_name)
    vector = model.encode([query], normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32)
