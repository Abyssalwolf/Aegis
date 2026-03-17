"""
Shared in-process state for the RAG API.

Singletons held here:
  - bm25:     in-memory BM25 sparse retriever, shared between ingest and query routes
  - qdrant:   single QdrantStore (single httpx.Client) for the whole process
  - embedder: single LocalEmbedder loaded once at startup; avoids a 20-30s stall
              on the first query while the 278 MB model is read from disk

Both qdrant and embedder use lazy initialisation so that imports don't cause
side-effects, but they are intentionally warmed up in the lifespan startup event.
"""

from core.retrieval.bm25_retriever import BM25Retriever

bm25 = BM25Retriever()

_qdrant = None
_embedder = None
_reranker = None


def get_qdrant():
    """Return the process-wide QdrantStore, creating it on first call."""
    global _qdrant
    if _qdrant is None:
        from stores.qdrant_store import QdrantStore
        _qdrant = QdrantStore()
    return _qdrant


def get_embedder():
    """Return the process-wide LocalEmbedder, loading the model on first call."""
    global _embedder
    if _embedder is None:
        from core.embeddings.local_embedder import LocalEmbedder
        _embedder = LocalEmbedder()
        _embedder._load()   # eagerly load weights into RAM now, not on first query
    return _embedder


def get_reranker():
    """Return the process-wide BGEReranker, loading the model on first call."""
    global _reranker
    if _reranker is None:
        from core.reranking.bge_reranker import BGEReranker
        _reranker = BGEReranker()
        _reranker._load()   # eagerly load cross-encoder weights into RAM
    return _reranker
