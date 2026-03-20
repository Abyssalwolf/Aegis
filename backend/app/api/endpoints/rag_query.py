"""
RAG query proxy endpoint.
POST /cases/{case_id}/query  — proxies to the RAG service, scoped to the case.
Forwards conversation history so the LLM can maintain multi-turn context.
"""
from typing import Any, List, Optional
from uuid import UUID
import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.api.endpoints.cases import check_case_access
from app.core.config import settings
from app.models.user import User
from app.models.case import Case

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    rewrite: bool = True
    messages: List[ChatMessage] = []


class SourceReference(BaseModel):
    index: int
    document_id: str
    source_path: str
    page_number: Optional[int] = None
    case_id: Optional[str] = None
    relevance_score: float
    chunk_type: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    queries_used: List[str]
    sources: List[SourceReference]
    chunks_retrieved: int
    chunks_after_rerank: int


@router.post("/{case_id}/query", response_model=QueryResponse)
async def query_case_documents(
    case_id: UUID,
    body: QueryRequest,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    Ask a question about the documents in a case.
    Validates case access, then proxies to the RAG service with the case_id filter.
    Forwards conversation history for multi-turn context.
    """
    result = await db.execute(select(Case).filter(Case.id == case_id))
    case = result.scalars().first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if not await check_case_access(db, case, current_user):
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.RAG_SERVICE_URL}/query/",
                json={
                    "query": body.query,
                    "case_id": str(case_id),
                    "top_k": body.top_k,
                    "rewrite": body.rewrite,
                    "stream": False,
                    "messages": [m.model_dump() for m in body.messages],
                },
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        logger.error(f"RAG query HTTP error for case {case_id}: {exc.response.text}")
        raise HTTPException(
            status_code=502,
            detail=f"RAG service returned an error: {exc.response.status_code}",
        )
    except httpx.RequestError as exc:
        logger.error(f"RAG service unreachable for case {case_id}: {exc}")
        raise HTTPException(
            status_code=503,
            detail="RAG service is currently unavailable. Please ensure it is running.",
        )
