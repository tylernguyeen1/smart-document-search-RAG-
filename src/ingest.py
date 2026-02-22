from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass
class Document:
    file_name: str
    source_path: str
    text: str


def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    raise ValueError(f"Unsupported file extension: {ext}")


def iter_supported_files(raw_dir: Path) -> Iterable[Path]:
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def load_documents(raw_dir: str | Path) -> list[Document]:
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw data directory not found: {root}")

    docs: list[Document] = []
    for path in iter_supported_files(root):
        text = extract_text(path)
        if not text:
            continue
        docs.append(
            Document(
                file_name=path.name,
                source_path=str(path.resolve()),
                text=text,
            )
        )
    return docs
