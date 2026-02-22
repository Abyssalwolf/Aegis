# Case File RAG System

A fully local Retrieval-Augmented Generation system for police case files.
No external API calls. Everything runs on your machine.

---

## Stack

| Component | Tool |
|---|---|
| PDF + Image parsing | Docling (CPU-native, IBM open source) |
| Embedder | `BAAI/bge-base-en-v1.5` via sentence-transformers |
| Vector store | Qdrant (Docker) |
| Sparse retrieval | `rank-bm25` |
| Reranker | `BAAI/bge-reranker-base` (cross-encoder) |
| Query rewriter + LLM | `qwen2.5:3b` via Ollama |
| API | FastAPI |

Designed for: Intel i7 1260P, 16GB RAM, integrated graphics (CPU-only).

---

## Setup

### 1. Prerequisites

```bash
# Docker (for Qdrant)
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:3b
ollama serve   # Run in a separate terminal
```

### 2. Python environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configuration

```bash
cp .env.example .env
# Edit .env if you need to change any defaults
```

### 4. Run the API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8080 --reload
```

API docs available at: http://localhost:8080/docs

---

## Usage

### Ingest a case file

```bash
curl -X POST http://localhost:8080/ingest/file \
  -F "file=@/path/to/case_report.pdf" \
  -F "case_id=CASE-2024-001" \
  -F "officer_id=OFFICER-42"
```

### Query the system

```bash
curl -X POST http://localhost:8080/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What witnesses were interviewed at the scene?",
    "case_id": "CASE-2024-001",
    "rewrite": true
  }'
```

Response includes `answer`, `sources` (with document + page references),
`queries_used` (original + rewrites), and retrieval stats.

---

## Architecture

```
User Query
    → Query Rewriter (Ollama qwen2.5:3b → 2 rewrites)
    → Hybrid Retriever (Qdrant dense + BM25 sparse, parallel)
    → RRF Fusion (merge ranked lists from all query variants)
    → BGE Reranker (cross-encoder, top-50 → top-5)
    → Context Builder (pack chunks with citations)
    → LLM (Ollama, grounded answer with [Source N] citations)
    → Response (answer + source list)
```

### File structure

```
rag-system/
├── config/settings.py           # All config via env vars
├── core/
│   ├── documents/
│   │   ├── models.py            # Chunk, DocumentRecord dataclasses
│   │   └── chunker.py           # Semantic chunker
│   ├── embeddings/
│   │   └── local_embedder.py    # BGE embedder
│   ├── retrieval/
│   │   ├── bm25_retriever.py    # Sparse retrieval
│   │   └── hybrid_retriever.py  # Dense + sparse + RRF
│   ├── reranking/
│   │   └── bge_reranker.py      # Cross-encoder reranker
│   └── generation/
│       └── llm_client.py        # Ollama client
├── ingestion/
│   ├── loaders/
│   │   ├── pdf_loader.py        # Docling PDF loader
│   │   └── image_loader.py      # Docling image/OCR loader
│   ├── processors/
│   │   └── cleaner.py           # Post-OCR text cleaning
│   └── pipeline.py              # Full ingestion orchestration
├── stores/
│   ├── qdrant_store.py          # Vector store (text + image collections)
│   └── document_store.py        # SQLite document tracker
├── query/
│   ├── query_rewriter.py        # Multi-query expansion
│   └── context_builder.py       # Prompt assembly with citations
└── api/
    ├── app.py                   # FastAPI app
    └── routes/
        ├── ingest.py            # /ingest endpoints
        └── query.py             # /query endpoint
```

---

## Memory budget (16GB RAM)

| Component | RAM |
|---|---|
| Docling (at ingest) | ~1.5 GB |
| BGE embedder | ~0.5 GB |
| BGE reranker | ~0.5 GB |
| qwen2.5:3b (4-bit) | ~2.5 GB |
| Qdrant + app | ~1.5 GB |
| **Total** | **~6.5 GB** |

Models are lazy-loaded and can be explicitly unloaded between
ingest and query phases to keep peak usage low.

---

## Important notes

- Every answer includes source citations linking to the exact document and page.
- The system will explicitly say when it cannot find relevant information.
- No data leaves your machine — all inference is local.
- The BM25 index is rebuilt after each ingestion batch. For large corpora
  this takes a few seconds.
