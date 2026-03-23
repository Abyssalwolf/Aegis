"""
Query route.
POST /query  — ask a question, get a cited answer

Full pipeline per request:
  query → rewrite → hybrid retrieve → rerank → build context → LLM → response
"""

import logging
from typing import Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from config.settings import settings
from core.retrieval.hybrid_retriever import HybridRetriever
from core.reranking.bge_reranker import BGEReranker
from core.generation.llm_client import OllamaClient
from query.query_rewriter import QueryRewriter
from query.context_builder import build_prompt, SYSTEM_PROMPT
from api.shared_state import bm25, get_qdrant, get_embedder, get_reranker

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Shared singletons, lazy-loaded ---
_embedder = None
_qdrant = None
_retriever: Optional[HybridRetriever] = None
_reranker: Optional[BGEReranker] = None
_rewriter: Optional[QueryRewriter] = None
_llm: Optional[OllamaClient] = None


def _get_retriever() -> HybridRetriever:
    global _embedder, _qdrant, _retriever
    if _retriever is None:
        _embedder = get_embedder()  # shared singleton, pre-loaded at startup
        _qdrant = get_qdrant()      # shared singleton — no second cold start
        _retriever = HybridRetriever(
            embedder=_embedder,
            qdrant=_qdrant,
            bm25=bm25,
        )
    return _retriever


def _get_reranker() -> BGEReranker:
    global _reranker
    if _reranker is None:
        _reranker = get_reranker()  # shared singleton, pre-loaded at startup
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


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    query: str
    case_id: Optional[str] = None
    top_k: int = settings.reranker_top_k
    rewrite: bool = True
    stream: bool = False
    # Prior turns (omit or send [] after "clear chat"). Used for the answer prompt only.
    chat_history: list[ChatMessage] = Field(default_factory=list)


class SourceReference(BaseModel):
    index: int
    document_id: str
    source_path: str
    page_number: Optional[int]
    case_id: Optional[str]
    relevance_score: float
    chunk_type: str
    display_name: Optional[str] = None
    evidence_category: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    reasoning: Optional[str] = None  # model chain-of-thought when enabled (Ollama think / tags)
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

    history_dicts = [m.model_dump() for m in request.chat_history]

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
        query=request.query,
        candidates=candidates,
        top_k=request.top_k,
    )
    logger.info(f"Reranked to {len(reranked)} chunks.")

    # 4. Build context + prompt
    prompt, source_dicts = build_prompt(
        request.query,
        reranked,
        chat_history=history_dicts,
    )

    # 5. Generate answer
    llm = _get_llm()
    out = llm.generate(
        prompt=prompt,
        system=SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=settings.llm_max_tokens,
        include_reasoning=True,
    )

    sources = [SourceReference(**s) for s in source_dicts]

    return QueryResponse(
        query=request.query,
        answer=out.content,
        reasoning=out.reasoning,
        queries_used=queries,
        sources=sources,
        chunks_retrieved=len(candidates),
        chunks_after_rerank=len(reranked),
    )
