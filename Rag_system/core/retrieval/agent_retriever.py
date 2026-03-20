"""
agent_retriever.py — RAG wrapper for document agents.

Uses your existing Qdrant store and embedder (via shared_state)
instead of ChromaDB. Each case gets its own Qdrant collection prefix.
"""

from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


def ingest_document(
    case_id: str,
    doc_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Chunk and embed a document into your existing Qdrant store.
    Reuses your existing ingestion pipeline so it lands in the same
    collection as everything else — no separate store needed.
    """
    try:
        from api.shared_state import get_qdrant, get_embedder
        from core.documents.chunker import chunk_text

        embedder = get_embedder()
        qdrant = get_qdrant()

        chunks = chunk_text(text)
        if not chunks:
            logger.warning(f"[{case_id}] No chunks produced for doc_id={doc_id}")
            return

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            vector = embedder.embed(chunk)
            qdrant.upsert(
                chunk_id=chunk_id,
                vector=vector,
                text=chunk,
                metadata={
                    **(metadata or {}),
                    "case_id": case_id,
                    "doc_id": doc_id,
                    "chunk_index": i,
                },
            )

        logger.info(f"[{case_id}] Ingested {len(chunks)} chunks for {doc_id}")

    except Exception as e:
        logger.error(f"[{case_id}] Ingestion failed: {e}", exc_info=True)


def query_rag(case_id: str, question: str, n_results: int = 5) -> str:
    """
    Query your existing Qdrant store filtered by case_id.
    Returns concatenated context string for the agent LLM call.
    """
    try:
        from api.shared_state import get_qdrant, get_embedder

        embedder = get_embedder()
        qdrant = get_qdrant()

        vector = embedder.embed(question)
        results = qdrant.search(
            vector=vector,
            top_k=n_results,
            filter={"case_id": case_id},
        )

        if not results:
            return ""

        context_parts = [
            f"[Result {i+1}]\n{r.text if hasattr(r, 'text') else str(r)}"
            for i, r in enumerate(results)
        ]
        return "\n\n".join(context_parts)

    except Exception as e:
        logger.error(f"[{case_id}] RAG query failed: {e}", exc_info=True)
        return ""