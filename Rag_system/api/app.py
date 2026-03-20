"""
FastAPI application.
Two primary route groups:
  - /ingest: upload and process PDF or image files
  - /query:  ask questions against the case file corpus
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.ingest import router as ingest_router
from api.routes.query import router as query_router
from api import shared_state
from config.settings import settings
from api.routes.agents import router as agents_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG system starting up...")

    # ── Qdrant ──────────────────────────────────────────────────────────────
    # Initialise the shared QdrantStore (single httpx.Client for the process)
    # and rebuild the BM25 index from whatever chunks are already in Qdrant.
    try:
        qdrant = shared_state.get_qdrant()
        pairs = qdrant.get_all_texts()
        if pairs:
            shared_state.bm25.build_index(pairs)
            logger.info(f"BM25 index rebuilt from Qdrant: {len(pairs)} chunks loaded.")
        else:
            logger.info("Qdrant collection is empty — BM25 index will be built after first ingestion.")
    except Exception as exc:
        logger.warning(f"Could not connect to Qdrant on startup: {exc}")

    # ── Embedder ─────────────────────────────────────────────────────────────
    # Pre-load the sentence-transformer weights into RAM now so that the first
    # query doesn't stall for 20-30 s waiting for a 278 MB model to load.
    try:
        shared_state.get_embedder()
        logger.info(f"Embedder pre-loaded: {settings.embedder_model}")
    except Exception as exc:
        logger.warning(f"Could not pre-load embedder on startup: {exc}")

    # ── Reranker ──────────────────────────────────────────────────────────────
    # The cross-encoder is loaded lazily on first query rather than at startup.
    # On Windows with a small page file, safe_open (used by safetensors) blocks
    # indefinitely instead of failing fast, which would hang the startup sequence.
    # The BGEReranker.rerank() method already has a graceful OSError fallback that
    # returns hybrid-retrieval results when the model cannot be loaded.
    # Fix the page file (System Properties → Advanced → Performance → Virtual Memory)
    # and the reranker will load normally on the first query.

    yield
    logger.info("RAG system shutting down.")


app = FastAPI(
    title="Case File RAG API",
    description="Retrieval-Augmented Generation over police case files.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router, prefix="/ingest", tags=["Ingestion"])
app.include_router(query_router, prefix="/query", tags=["Query"])
app.include_router(agents_router, prefix="/agents", tags=["Agents"])

@app.get("/health")
def health():
    return {"status": "ok", "model": settings.ollama_model}
