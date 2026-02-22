from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.index import draft_answer, search


def main() -> None:
    parser = argparse.ArgumentParser(description="Query local FAISS index")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--index-dir", default="data/index", help="Index folder")
    parser.add_argument("--top-k", type=int, default=5, help="Top results")
    args = parser.parse_args()

    results = search(query=args.query, index_dir=args.index_dir, top_k=args.top_k)
    print("\nAnswer:")
    print(draft_answer(args.query, results))

    print("\nTop passages:")
    if not results:
        print("No results.")
        return

    for rank, item in enumerate(results, start=1):
        snippet = item["text"][:240].replace("\n", " ")
        print(
            f"{rank}. score={item['score']:.4f} | file={item['file_name']} | "
            f"chunk={item['chunk_id']}\n   {snippet}"
        )


if __name__ == "__main__":
    main()
