# Smart Document Search / RAG

Local document search with embeddings + FAISS, supporting `.pdf`, `.txt`, `.md`.

## MVP Features
- Ingest documents from `data/raw`
- Extract text (`pypdf` for PDF)
- Chunk with overlap
- Embed chunks with `sentence-transformers`
- Store/search vectors in FAISS
- Return top-k passages with citations (`file_name`, `chunk_id`)
- CLI scripts for indexing and querying
- Streamlit UI for demo

## Project Layout
```text
smart-document-search-RAG-/
  data/
    raw/                 # put PDFs/text/markdown here
    index/               # generated FAISS + metadata (gitignored)
  src/
    ingest.py
    chunk.py
    embed.py
    index.py
    api.py
    ui_streamlit.py
  scripts/
    build_index.py
    query.py
  requirements.txt
  README.md
  .gitignore
```

## Environment Setup (Windows PowerShell)
```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run
1. Put files in `data/raw`.
2. Build index:
```powershell
python scripts/build_index.py --raw-dir data/raw --index-dir data/index
```
3. Query from CLI:
```powershell
python scripts/query.py "What does the contract say about termination?" --index-dir data/index --top-k 5
```
4. Streamlit UI:
```powershell
streamlit run src/ui_streamlit.py
```
5. FastAPI (optional):
```powershell
uvicorn src.api:app --reload
```

## Minimal React Web App (Drag/Drop + Ask)
Backend:
```powershell
uvicorn src.api:app --reload
```

Frontend (new terminal):
```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

Flow:
1. Drag/drop a PDF (or choose file).
2. Click `Upload and Build Index`.
3. Ask a question.
4. Read summary + citations.

## Notes on PyTorch
- Yes, this uses PyTorch indirectly through `sentence-transformers`.
- You are running pretrained models for inference only (no training code required for MVP).

## Resume-Grade Next Steps
- Add reranking (cross-encoder on top FAISS top-20)
- Add evaluation script (Recall@k, MRR, nDCG)
- Add incremental indexing by file hash/mtime
- Compare FAISS-only vs FAISS+rerank in README
