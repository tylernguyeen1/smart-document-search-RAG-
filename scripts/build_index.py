from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.index import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from data/raw")
    parser.add_argument("--raw-dir", default="data/raw", help="Input documents folder")
    parser.add_argument("--index-dir", default="data/index", help="Output index folder")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()

    metadata = build_index(
        raw_dir=args.raw_dir,
        index_dir=args.index_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print("Index build complete")
    print(f"Documents and chunks indexed: {metadata['count']}")
    print(f"Model: {metadata['model_name']}")


if __name__ == "__main__":
    main()
