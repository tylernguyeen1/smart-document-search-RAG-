from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.index import build_index, draft_answer, search


def main() -> None:
    st.set_page_config(page_title="Smart Document Search (RAG)", layout="wide")
    st.title("Smart Document Search / RAG")

    raw_dir = st.text_input("Document folder", value="data/raw")
    index_dir = st.text_input("Index folder", value="data/index")

    cols = st.columns([1, 1, 2])
    with cols[0]:
        chunk_size = st.number_input("Chunk size", value=1000, min_value=300, max_value=3000)
    with cols[1]:
        overlap = st.number_input("Overlap", value=200, min_value=0, max_value=1000)

    if st.button("Build / Refresh Index", type="primary"):
        with st.spinner("Building index..."):
            meta = build_index(raw_dir=raw_dir, index_dir=index_dir, chunk_size=int(chunk_size), overlap=int(overlap))
        st.success(f"Indexed {meta['count']} chunks with {meta['model_name']}")

    query = st.text_input("Ask a question")
    top_k = st.slider("Top-k", min_value=1, max_value=20, value=5)

    if st.button("Search"):
        if not query.strip():
            st.warning("Enter a query first.")
            return
        if not (Path(index_dir) / "faiss.index").exists():
            st.error("Index not found. Build the index first.")
            return

        results = search(query=query, index_dir=index_dir, top_k=top_k)
        st.subheader("Answer")
        st.write(draft_answer(query, results))

        st.subheader("Top Passages")
        if not results:
            st.info("No results found.")
            return

        for i, item in enumerate(results, start=1):
            st.markdown(
                f"**{i}. {item['file_name']} ({item['chunk_id']})**  \n"
                f"`score={item['score']:.4f}`"
            )
            st.write(item["text"])


if __name__ == "__main__":
    main()
