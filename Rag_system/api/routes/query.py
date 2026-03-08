"""
Query route.
POST /query  — ask a question, get a cited answer

Full pipeline per request:
  query → rewrite → hybrid retrieve → rerank → build context → LLM → response
"""

import logging
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config.settings import settings
from core.embeddings.local_embedder import LocalEmbedder
from core.retrieval.bm25_retriever import BM25Retriever
from core.retrieval.hybrid_retriever import HybridRetriever
from core.reranking.bge_reranker import BGEReranker
from core.generation.llm_client import OllamaClient
from query.query_rewriter import QueryRewriter
from query.context_builder import build_prompt, SYSTEM_PROMPT
from stores.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Shared singletons, lazy-loaded ---
_embedder: Optional[LocalEmbedder] = None
_qdrant: Optional[QdrantStore] = None
_bm25: Optional[BM25Retriever] = None
_retriever: Optional[HybridRetriever] = None
_reranker: Optional[BGEReranker] = None
_rewriter: Optional[QueryRewriter] = None
_llm: Optional[OllamaClient] = None


def _get_retriever() -> HybridRetriever:
    global _embedder, _qdrant, _bm25, _retriever
    if _retriever is None:
        _embedder = LocalEmbedder()
        _qdrant = QdrantStore()
        _bm25 = BM25Retriever()
        _retriever = HybridRetriever(
            embedder=_embedder,
            qdrant=_qdrant,
            bm25=_bm25,
        )
    return _retriever


def _get_reranker() -> BGEReranker:
    global _reranker
    if _reranker is None:
        _reranker = BGEReranker()
    return _reranker


def _get_rewriter() -> QueryRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter


def _get_llm() -> OllamaClient:
    global _llm
    if _llm is None:
        _llm = OllamaClient()
    return _llm


# --- Request / Response models ---

class QueryRequest(BaseModel):
    query: str
    case_id: Optional[str] = None       # Scope retrieval to a specific case
    top_k: int = settings.reranker_top_k
    rewrite: bool = True                # Enable/disable query rewriting
    stream: bool = False


class SourceReference(BaseModel):
    index: int
    document_id: str
    source_path: str
    page_number: Optional[int]
    case_id: Optional[str]
    relevance_score: float
    chunk_type: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    queries_used: list[str]
    sources: list[SourceReference]
    chunks_retrieved: int
    chunks_after_rerank: int


# --- Route ---

@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Answer a question from case file documents.
    Returns a cited answer with source references.
    """
    logger.info(f"Query received: '{request.query}' (case_id={request.case_id})")

    # 1. Query rewriting
    rewriter = _get_rewriter()
    if request.rewrite:
        queries = rewriter.rewrite(request.query)
    else:
        queries = [request.query]

    logger.info(f"Queries after rewriting: {queries}")

    # 2. Hybrid retrieval (multi-query)
    retriever = _get_retriever()
    candidates = retriever.search_multi_query(
        queries=queries,
        top_k=settings.retrieval_top_k,
        case_id=request.case_id,
    )
    logger.info(f"Retrieved {len(candidates)} candidates before reranking.")

    if not candidates:
        return QueryResponse(
            query=request.query,
            answer="No relevant documents found for this query. "
                   "Please ensure the relevant case files have been ingested.",
            queries_used=queries,
            sources=[],
            chunks_retrieved=0,
            chunks_after_rerank=0,
        )

    # 3. Reranking
    reranker = _get_reranker()
    reranked = reranker.rerank(
        query=request.query,    # Rerank against original query for precision
        candidates=candidates,
        top_k=request.top_k,
    )
    logger.info(f"Reranked to {len(reranked)} chunks.")

    # 4. Build context + prompt
    prompt, source_dicts = build_prompt(request.query, reranked)

    # 5. Generate answer
    llm = _get_llm()
    answer = llm.generate(
        prompt=prompt,
        system=SYSTEM_PROMPT,
        temperature=0.1,    # Low temp — factual, grounded answers only
        max_tokens=1024,
    )

    sources = [SourceReference(**s) for s in source_dicts]

    return QueryResponse(
        query=request.query,
        answer=answer,
        queries_used=queries,
        sources=sources,
        chunks_retrieved=len(candidates),
        chunks_after_rerank=len(reranked),
    )
