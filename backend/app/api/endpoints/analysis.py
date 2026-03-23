from __future__ import annotations

import json
import logging
from typing import Any, List
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.api.endpoints.cases import check_case_access
from app.core.config import settings
from app.core.insights_case_id import uuid_to_insights_case_id
from app.models.analysis import CaseAnalysis
from app.models.case import Case
from app.models.user import User
from app.schemas.analysis import CaseAnalysisOut

logger = logging.getLogger(__name__)

router = APIRouter()


def _agents_base() -> str:
    return f"{settings.RAG_SERVICE_URL.rstrip('/')}/agents"

# Cap stored snapshot size (Postgres text is large; keep responses reasonable).
_MAX_SNAPSHOT_CHARS = 250_000


def _slim_blackboard(board: Any) -> dict[str, Any]:
    if not isinstance(board, dict):
        return {}
    keys = ("messages", "anomalies", "findings", "insights", "status", "case_id")
    return {k: board[k] for k in keys if k in board}


async def _fetch_insights_snapshot_text(rag_case_id: int) -> str:
    """Pull supervisor brief + blackboard JSON from RAG (no auth — internal service)."""
    base = f"{_agents_base()}/cases/{rag_case_id}"
    brief_text = ""
    board: dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            br = await client.get(f"{base}/blackboard/brief")
            if br.is_success:
                data = br.json()
                if isinstance(data, dict) and isinstance(data.get("brief"), str):
                    brief_text = data["brief"]
            bb = await client.get(f"{base}/blackboard")
            if bb.is_success:
                payload = bb.json()
                if isinstance(payload, dict):
                    board = payload
    except httpx.HTTPError as exc:
        logger.warning("RAG snapshot fetch HTTP error: %s", exc)
    except Exception as exc:
        logger.warning("RAG snapshot fetch failed: %s", exc)

    slim = _slim_blackboard(board)
    body = (
        "=== Supervisor brief (markdown) ===\n\n"
        f"{brief_text}\n\n"
        "=== Blackboard snapshot (JSON subset) ===\n"
        f"{json.dumps(slim, indent=2, default=str)}"
    )
    if len(body) > _MAX_SNAPSHOT_CHARS:
        body = body[:_MAX_SNAPSHOT_CHARS] + "\n\n[truncated]"
    return body


@router.post("/{case_id}/analysis", response_model=CaseAnalysisOut)
async def request_analysis(
    case_id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    result = await db.execute(select(Case).filter(Case.id == case_id))
    case = result.scalars().first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if not await check_case_access(db, case, current_user):
        raise HTTPException(status_code=403, detail="Forbidden")

    rag_id = uuid_to_insights_case_id(case_id)
    result_text = await _fetch_insights_snapshot_text(rag_id)

    row = CaseAnalysis(
        case_id=case_id,
        analysis_type="insights_snapshot",
        result_text=result_text,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


@router.get("/{case_id}/analysis", response_model=List[CaseAnalysisOut])
async def get_analysis(
    case_id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    result = await db.execute(select(Case).filter(Case.id == case_id))
    case = result.scalars().first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if not await check_case_access(db, case, current_user):
        raise HTTPException(status_code=403, detail="Forbidden")

    q = await db.execute(
        select(CaseAnalysis)
        .filter(CaseAnalysis.case_id == case_id)
        .order_by(CaseAnalysis.created_at.desc())
    )
    return q.scalars().all()
