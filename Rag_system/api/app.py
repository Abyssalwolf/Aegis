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
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG system starting up...")
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


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.ollama_model}
