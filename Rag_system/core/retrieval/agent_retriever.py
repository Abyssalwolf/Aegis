"""
agent_retriever.py — RAG wrapper for document agents.

Uses your existing IngestionPipeline (now supports .txt files)
and HybridRetriever. Works in both FastAPI and Celery processes.
"""

from __future__ import annotations
import logging
import os
import tempfile
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)


def _get_qdrant():
    from stores.qdrant_store import QdrantStore
    return QdrantStore()


def ingest_document(
    case_id: str,
    doc_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Ingest extracted text into Qdrant via the existing IngestionPipeline.
    Writes to a temp .txt file so the pipeline can process it.
    """
    try:
        from ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline(qdrant=_get_qdrant())

        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        try:
            pipeline.ingest_file(
                file_path=tmp_path,
                case_id=case_id,
                officer_id=None,
            )
            logger.info(f"[{case_id}] Ingested document {doc_id}")
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"[{case_id}] Ingestion failed: {e}", exc_info=True)


def query_rag(case_id: str, question: str, n_results: int = 5) -> str:
    """
    Query Qdrant for relevant chunks filtered by case_id.
    Returns concatenated context string for the agent LLM call.
    """
    try:
        from core.retrieval.hybrid_retriever import HybridRetriever
        from core.retrieval.bm25_retriever import BM25Retriever
        from core.embeddings.local_embedder import LocalEmbedder

        embedder = LocalEmbedder()
        qdrant = _get_qdrant()
        bm25 = BM25Retriever()

        pairs = qdrant.get_all_texts()
        if pairs:
            bm25.build_index(pairs)

        retriever = HybridRetriever(
            embedder=embedder,
            qdrant=qdrant,
            bm25=bm25,
        )

        results = retriever.search_multi_query(
            queries=[question],
            top_k=n_results,
            case_id=case_id,
        )

        if not results:
            return ""

        context_parts = [
            f"[Result {i+1}]\n{r.chunk.text if hasattr(r, 'chunk') else str(r)}"
            for i, r in enumerate(results[:n_results])
        ]
        return "\n\n".join(context_parts)

    except Exception as e:
        logger.error(f"[{case_id}] RAG query failed: {e}", exc_info=True)
        return ""
