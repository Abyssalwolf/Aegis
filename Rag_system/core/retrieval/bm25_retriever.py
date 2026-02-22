"""
BM25 sparse retriever.
Builds an in-memory BM25 index over all stored chunk texts.
Index is persisted to disk and rebuilt only when new documents are ingested.
"""

import pickle
import logging
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from config.settings import settings
from core.documents.models import Chunk, ChunkType, RetrievedChunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


class BM25Retriever:

    def __init__(self, index_path: Path = settings.bm25_index_path):
        self.index_path = index_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self._bm25: Optional[BM25Okapi] = None
        self._chunk_ids: list[str] = []
        self._chunk_texts: list[str] = []

        if self.index_path.exists():
            self._load_index()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self, chunk_id_text_pairs: list[tuple[str, str]]) -> None:
        """
        Build BM25 index from (chunk_id, text) pairs.
        Call this after each ingestion batch.
        """
        if not chunk_id_text_pairs:
            logger.warning("No chunks provided to build BM25 index.")
            return

        self._chunk_ids = [p[0] for p in chunk_id_text_pairs]
        self._chunk_texts = [p[1] for p in chunk_id_text_pairs]

        tokenized_corpus = [_tokenize(text) for text in self._chunk_texts]
        self._bm25 = BM25Okapi(tokenized_corpus)

        self._save_index()
        logger.info(f"BM25 index built with {len(self._chunk_ids)} chunks.")

    def update_index(self, new_pairs: list[tuple[str, str]]) -> None:
        """
        Add new chunks to the existing index.
        BM25Okapi doesn't support incremental updates, so we rebuild.
        For large corpora, this could be optimized with chunked persistence.
        """
        existing_pairs = list(zip(self._chunk_ids, self._chunk_texts))
        all_pairs = existing_pairs + new_pairs
        self.build_index(all_pairs)

    def _save_index(self) -> None:
        data = {
            "chunk_ids": self._chunk_ids,
            "chunk_texts": self._chunk_texts,
            "bm25": self._bm25,
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)
        logger.debug(f"BM25 index saved to {self.index_path}.")

    def _load_index(self) -> None:
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self._chunk_ids = data["chunk_ids"]
            self._chunk_texts = data["chunk_texts"]
            self._bm25 = data["bm25"]
            logger.info(f"BM25 index loaded: {len(self._chunk_ids)} chunks.")
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {e}. Will rebuild on next ingest.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = settings.retrieval_top_k) -> list[RetrievedChunk]:
        """
        Return top_k chunks by BM25 score.
        Returns RetrievedChunk with text populated but no embedding.
        """
        if self._bm25 is None or not self._chunk_ids:
            logger.warning("BM25 index is empty. Returning no results.")
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score descending
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results: list[RetrievedChunk] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue    # No match at all — skip

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
