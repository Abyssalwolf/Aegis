"""
BGE cross-encoder reranker.
Takes top-K retrieved chunks and reranks them using BAAI/bge-reranker-base.

The cross-encoder sees the query and each passage together, producing a
relevance score that's far more accurate than cosine similarity — at the
cost of O(N) inference passes. We only run this on a small candidate set
(top-50 from hybrid retrieval → rerank to top-5).
"""

import logging
from typing import Optional

from sentence_transformers import CrossEncoder

from config.settings import settings
from core.documents.models import RetrievedChunk

logger = logging.getLogger(__name__)


class BGEReranker:

    def __init__(
        self,
        model_name: str = settings.reranker_model,
        device: str = settings.embedder_device,
        top_k: int = settings.reranker_top_k,
    ):
        self.model_name = model_name
        self.device = device
        self.top_k = top_k
        self._model: Optional[CrossEncoder] = None

    def _load(self) -> CrossEncoder:
        if self._model is None:
            logger.info(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512,
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Rerank candidates by cross-encoder relevance score.
        Returns top_k results sorted by reranker score (descending).
        """
        if not candidates:
            return []

        k = top_k or self.top_k
        model = self._load()

        # Pair query with each candidate text
        pairs = [[query, rc.chunk.text] for rc in candidates]

        logger.debug(f"Reranking {len(pairs)} candidates with cross-encoder.")
        scores = model.predict(pairs, show_progress_bar=False)

        # Attach reranker scores and sort
        scored: list[tuple[float, RetrievedChunk]] = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[RetrievedChunk] = []
        for score, retrieved in scored[:k]:
            results.append(RetrievedChunk(
                chunk=retrieved.chunk,
                score=float(score),
                retrieval_method="reranked",
            ))

        logger.info(f"Reranked {len(candidates)} → {len(results)} chunks.")
        return results

    def unload(self) -> None:
        """Free reranker from memory after query."""
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
