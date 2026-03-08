"""
Hybrid retriever.
Runs dense (Qdrant) and sparse (BM25) retrieval in parallel,
then merges results using Reciprocal Rank Fusion (RRF).

RRF formula: score(d) = sum(1 / (k + rank(d)))
where k=60 is a smoothing constant (standard value from the original paper).
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from config.settings import settings
from core.documents.models import ChunkType, RetrievedChunk
from core.embeddings.local_embedder import LocalEmbedder
from core.retrieval.bm25_retriever import BM25Retriever
from stores.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

RRF_K = 60  # Standard RRF smoothing constant


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank + 1)    # +1 because ranks are 0-indexed


class HybridRetriever:

    def __init__(
        self,
        embedder: LocalEmbedder,
        qdrant: QdrantStore,
        bm25: BM25Retriever,
    ):
        self.embedder = embedder
        self.qdrant = qdrant
        self.bm25 = bm25

    def search(
        self,
        query: str,
        top_k: int = settings.retrieval_top_k,
        case_id: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Run dense + sparse retrieval in parallel and fuse with RRF.
        Returns up to top_k results sorted by RRF score.
        """
        query_embedding = self.embedder.encode_single(query).tolist()

        # Run both retrievers in parallel
        dense_results: list[RetrievedChunk] = []
        sparse_results: list[RetrievedChunk] = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(
                self.qdrant.search,
                query_embedding=query_embedding,
                top_k=top_k,
                chunk_type=ChunkType.TEXT,
                case_id=case_id,
            )
            sparse_future = executor.submit(
                self.bm25.search,
                query=query,
                top_k=top_k,
            )

            for future in as_completed([dense_future, sparse_future]):
                if future == dense_future:
                    dense_results = future.result()
                else:
                    sparse_results = future.result()

        logger.debug(
            f"Dense: {len(dense_results)} results, Sparse: {len(sparse_results)} results."
        )

        fused = self._rrf_fusion(dense_results, sparse_results, top_k=top_k)

        # Enrich fused results with full payload from Qdrant if sparse-only
        fused = self._enrich_sparse_results(fused, query_embedding, case_id)

        return fused

    def search_multi_query(
        self,
        queries: list[str],
        top_k: int = settings.retrieval_top_k,
        case_id: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Run hybrid search for multiple query variants (from query rewriter),
        then fuse all results together.
        """
        all_ranked_lists: list[list[RetrievedChunk]] = []
        for query in queries:
            results = self.search(query, top_k=top_k, case_id=case_id)
            all_ranked_lists.append(results)

        return self._multi_list_rrf_fusion(all_ranked_lists, top_k=top_k)

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    def _rrf_fusion(
        self,
        dense: list[RetrievedChunk],
        sparse: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        return self._multi_list_rrf_fusion([dense, sparse], top_k=top_k)

    def _multi_list_rrf_fusion(
        self,
        ranked_lists: list[list[RetrievedChunk]],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Fuse multiple ranked lists using Reciprocal Rank Fusion."""
        # chunk_id → (accumulated_rrf_score, RetrievedChunk)
        scores: dict[str, float] = {}
        chunks: dict[str, RetrievedChunk] = {}

        for ranked_list in ranked_lists:
            for rank, retrieved in enumerate(ranked_list):
                cid = retrieved.chunk.chunk_id
                rrf = _rrf_score(rank)
                scores[cid] = scores.get(cid, 0.0) + rrf
                # Prefer dense results (have full payload) over sparse
                if cid not in chunks or retrieved.retrieval_method == "dense":
                    chunks[cid] = retrieved

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        results: list[RetrievedChunk] = []
        for cid in sorted_ids[:top_k]:
            retrieved = chunks[cid]
            results.append(RetrievedChunk(
                chunk=retrieved.chunk,
                score=scores[cid],
                retrieval_method="hybrid",
            ))

        return results

    def _enrich_sparse_results(
        self,
        results: list[RetrievedChunk],
        query_embedding: list[float],
        case_id: Optional[str],
    ) -> list[RetrievedChunk]:
        """
        Sparse results only have chunk_id + text, missing payload metadata.
        For those, we do a Qdrant point lookup to fill in the full payload.
        This is a best-effort enrichment; if lookup fails we keep as-is.
        """
        # For now, text from BM25 is sufficient for reranking.
        # Full metadata enrichment can be added via qdrant retrieve() if needed.
        return results
