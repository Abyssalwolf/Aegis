"""
BM25 sparse retriever.
Builds an in-memory BM25 index over all stored chunk texts.
Index is rebuilt from Qdrant on RAG service startup — no local file persistence.
This ensures correctness across restarts and machine migrations.
"""

import logging
from typing import Optional

from rank_bm25 import BM25Okapi

from config.settings import settings
from core.documents.models import Chunk, ChunkType, RetrievedChunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


class BM25Retriever:

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._chunk_ids: list[str] = []
        self._chunk_texts: list[str] = []

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self, chunk_id_text_pairs: list[tuple[str, str]]) -> None:
        """
        Build BM25 index from (chunk_id, text) pairs.
        Called once on startup (from Qdrant) and after each ingestion batch.
        """
        if not chunk_id_text_pairs:
            logger.warning("No chunks provided to build BM25 index — index will be empty.")
            return

        self._chunk_ids = [p[0] for p in chunk_id_text_pairs]
        self._chunk_texts = [p[1] for p in chunk_id_text_pairs]

        tokenized_corpus = [_tokenize(text) for text in self._chunk_texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built in-memory with {len(self._chunk_ids)} chunks.")

    def update_index(self, new_pairs: list[tuple[str, str]]) -> None:
        """
        Incrementally add new chunks to the existing index.
        BM25Okapi doesn't support true incremental updates, so we rebuild from scratch.
        """
        existing_pairs = list(zip(self._chunk_ids, self._chunk_texts))
        all_pairs = existing_pairs + new_pairs
        self.build_index(all_pairs)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = settings.retrieval_top_k) -> list[RetrievedChunk]:
        """Return top_k chunks by BM25 score."""
        if self._bm25 is None or not self._chunk_ids:
            logger.warning("BM25 index is empty. Returning no results.")
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results: list[RetrievedChunk] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue

            chunk = Chunk(
                chunk_id=self._chunk_ids[idx],
                chunk_type=ChunkType.TEXT,
                text=self._chunk_texts[idx],
            )
            results.append(RetrievedChunk(
                chunk=chunk,
                score=float(scores[idx]),
                retrieval_method="sparse",
            ))

        return results

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and len(self._chunk_ids) > 0
