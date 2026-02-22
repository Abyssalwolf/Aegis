# Complete Setup Guide — Case File RAG System
> Intel i7 1260P · 16GB RAM · Windows/Linux · CPU-only

---

## Prerequisites — Install These First

### 1. Python 3.11
Check if you have it:
```bash
python --version
```
If not, download from https://www.python.org/downloads/  
**Important on Windows:** Check "Add Python to PATH" during install.

---

### 2. Docker Desktop
Used to run Qdrant (the vector database) as a container.  
Download: https://www.docker.com/products/docker-desktop  
After installing, open Docker Desktop and let it finish starting up before continuing.

---

### 3. Git
Check if you have it:
```bash
git --version
```
If not: https://git-scm.com/downloads

---

### 4. Ollama (local LLM runner)
**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```
**Windows:**  
Download the installer from https://ollama.ai/download

Verify it installed:
```bash
ollama --version
```

Pull the LLM model we'll use for query rewriting and answer generation (~2GB download):
```bash
ollama pull qwen2.5:3b
```

---

## Project Setup

### 5. Create your project folder and virtual environment

**Linux / macOS:**
```bash
mkdir case-rag && cd case-rag
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
mkdir case-rag; cd case-rag
python -m venv .venv
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of your terminal prompt. Every command from here on assumes the venv is active.

---

### 6. Drop in the project files

Copy the entire `rag-system/` folder (from the files you downloaded) into your `case-rag/` directory. Your structure should look like:

```
case-rag/
├── .venv/
└── rag-system/
    ├── api/
    ├── config/
    ├── core/
    ├── ingestion/
    ├── query/
    ├── stores/
    ├── requirements.txt
    ├── .env.example
    └── README.md
```

Move into the project root:
```bash
cd rag-system
```

---

### 7. Install Python dependencies

This will take a few minutes — PyTorch and Docling are large:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**If you get a PyTorch error on Windows**, install it manually first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Verify the key packages installed correctly:
```bash
python -c "import torch; print('torch:', torch.__version__)"
python -c "import sentence_transformers; print('sentence-transformers: ok')"
python -c "import docling; print('docling: ok')"
python -c "import qdrant_client; print('qdrant-client: ok')"
```

---

### 8. Download the embedding and reranking models

These are downloaded from HuggingFace on first use, but it's better to
pre-download them now so there are no surprises at runtime:

```bash
python - <<'EOF'
from sentence_transformers import SentenceTransformer, CrossEncoder
print("Downloading embedder (~400MB)...")
SentenceTransformer("BAAI/bge-base-en-v1.5")
print("Downloading reranker (~400MB)...")
CrossEncoder("BAAI/bge-reranker-base")
print("All models downloaded.")
EOF
```

Models are cached in `~/.cache/huggingface/` — you only need to do this once.

---

### 9. Configure your environment

```bash
cp .env.example .env
```

The defaults in `.env` work out of the box. Only change things if you need to:
- Point to a different Qdrant host
- Use a different Ollama model
- Change chunk sizes

---

## Starting the Services

You need **three things running** every time you use the system.  
Open three separate terminal windows/tabs for this.

### Terminal 1 — Qdrant (vector database)
```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v $(pwd)/data/qdrant:/qdrant/storage \
  qdrant/qdrant
```

**Windows PowerShell:**
```powershell
docker run -d `
  --name qdrant `
  -p 6333:6333 `
  -v ${PWD}/data/qdrant:/qdrant/storage `
  qdrant/qdrant
```

The `-v` flag persists your vector data to disk so it survives restarts.

Check it's running:
```bash
curl http://localhost:6333/healthz
# Should return: {"title":"qdrant - vector search engine"}
```

> **Next time you start:** If the container already exists, use:
> `docker start qdrant`

---

### Terminal 2 — Ollama (local LLM)
```bash
ollama serve
```

Leave this running. You should see: `Listening on 127.0.0.1:11434`

Check it's working:
```bash
curl http://localhost:11434/api/tags
# Should list your downloaded models including qwen2.5:3b
```

---

### Terminal 3 — The RAG API
Make sure your venv is active, then from inside `rag-system/`:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8080 --reload
```

You should see:
```
INFO: RAG system starting up...
INFO: Uvicorn running on http://0.0.0.0:8080
```

Open http://localhost:8080/docs in your browser — this is the interactive
API documentation where you can test everything.

---

## Smoke Test — Verify Everything Works

### Test 1: Health check
```bash
curl http://localhost:8080/health
# Expected: {"status":"ok","model":"qwen2.5:3b"}
```

### Test 2: Ingest a PDF
```bash
curl -X POST http://localhost:8080/ingest/file \
  -F "file=@/path/to/your/test.pdf" \
  -F "case_id=TEST-001"
```

Expected response:
```json
{
  "document_id": "...",
  "filename": "test.pdf",
  "status": "completed",
  "chunk_count": 12
}
```

### Test 3: Query it
```bash
curl -X POST http://localhost:8080/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "case_id": "TEST-001",
    "rewrite": true
  }'
```

Expected response includes `answer`, `sources`, and `queries_used`.

---

## Troubleshooting

**`ModuleNotFoundError` when starting uvicorn**  
Make sure you're running from inside the `rag-system/` directory with your venv active.

**`Cannot reach Ollama` error**  
Ollama isn't running. Start it with `ollama serve` in a separate terminal.

**Qdrant connection refused**  
Docker container isn't running. Run `docker start qdrant` or the full `docker run` command above.

**Docling is very slow on first PDF**  
Normal — it downloads some additional models on first use (~200MB). Subsequent runs are faster.

**`torch not found` or BLAS warnings**  
These are harmless warnings on CPU. If torch isn't found, re-run:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`

**Out of memory during ingestion**  
Reduce batch size in `ingestion/pipeline.py`, change `batch_size=32` to `batch_size=8`.

---

## Everyday Workflow

Once everything is set up, your startup sequence each time is:

```bash
# 1. Start Qdrant (if not already running)
docker start qdrant

# 2. Start Ollama (new terminal)
ollama serve

# 3. Activate venv and start API (new terminal)
cd case-rag/rag-system
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
uvicorn api.app:app --host 0.0.0.0 --port 8080 --reload
```

Then use http://localhost:8080/docs to interact with the system.
