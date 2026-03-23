"""
Insights / multi-agent proxy.
Forwards to RAG service `GET/POST /agents/...` with integer case_id derived from Case UUID.

Frontend should call:
  /api/v1/cases/{uuid}/insights/...
not the RAG host directly.
"""
from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Optional
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.api.endpoints.cases import check_case_access
from app.core.config import settings
from app.core.insights_case_id import uuid_to_insights_case_id
from app.models.case import Case
from app.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter()

VALID_FILE_TYPES = frozenset(
    {
        "fir",
        "case_diary",
        "statement",
        "scene_of_crime",
        "forensic",
        "seizure",
        "arrest_remand",
    }
)


def _agents_base() -> str:
    return f"{settings.RAG_SERVICE_URL.rstrip('/')}/agents"


async def _get_case_and_rag_id(
    case_id: UUID,
    db: AsyncSession,
    current_user: User,
) -> tuple[Case, int]:
    result = await db.execute(select(Case).filter(Case.id == case_id))
    case = result.scalars().first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    if not await check_case_access(db, case, current_user):
        raise HTTPException(status_code=403, detail="Forbidden")
    rag_case_id = uuid_to_insights_case_id(case.id)
    return case, rag_case_id


def _rewrite_agent_links(payload: Any, case_uuid: UUID) -> Any:
    """Turn RAG-relative /agents/... URLs into backend insights paths where helpful."""
    if not isinstance(payload, dict):
        return payload
    base = f"{settings.API_V1_STR}/cases/{case_uuid}/insights"
    for key in ("poll_url", "blackboard_url", "stream_url"):
        if key not in payload or not isinstance(payload[key], str):
            continue
        v = payload[key]
        if v.startswith("/agents/tasks/"):
            tid = v.removeprefix("/agents/tasks/")
            payload[key] = f"{base}/tasks/{tid}"
        elif v.startswith("/agents/cases/") and "/blackboard" in v:
            payload[key] = f"{base}/blackboard"
        elif v.startswith("/agents/cases/") and "/stream" in v:
            payload[key] = f"{base}/stream"
    return payload


# --- Blackboard ---


@router.get("/{case_id}/insights/blackboard")
async def insights_blackboard(
    case_id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    _, rag_id = await _get_case_and_rag_id(case_id, db, current_user)
    url = f"{_agents_base()}/cases/{rag_id}/blackboard"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as exc:
        logger.error("RAG insights blackboard HTTP %s: %s", exc.response.status_code, exc.response.text)
        raise HTTPException(status_code=502, detail="RAG insights service returned an error")
    except httpx.RequestError as exc:
        logger.error("RAG insights unreachable: %s", exc)
        raise HTTPException(status_code=503, detail="RAG service unavailable")


@router.get("/{case_id}/insights/blackboard/brief")
async def insights_blackboard_brief(
    case_id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    _, rag_id = await _get_case_and_rag_id(case_id, db, current_user)
    url = f"{_agents_base()}/cases/{rag_id}/blackboard/brief"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError:
        raise HTTPException(status_code=502, detail="RAG insights service returned an error")
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="RAG service unavailable")


@router.get("/{case_id}/insights/stream")
async def insights_stream(
    case_id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
):
    _, rag_id = await _get_case_and_rag_id(case_id, db, current_user)
    url = f"{_agents_base()}/cases/{rag_id}/stream"

    async def gen():
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        yield chunk
        except httpx.HTTPError as exc:
            logger.error("Insights stream error: %s", exc)
            yield b"data: " + json.dumps({"error": "stream_failed"}).encode() + b"\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# --- Supervisor & tasks ---


@router.post("/{case_id}/insights/report")
async def insights_report(
    case_id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    _, rag_id = await _get_case_and_rag_id(case_id, db, current_user)
    url = f"{_agents_base()}/cases/{rag_id}/report"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url)
            r.raise_for_status()
            data = r.json()
            return _rewrite_agent_links(data, case_id)
    except httpx.HTTPStatusError:
        raise HTTPException(status_code=502, detail="RAG insights service returned an error")
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="RAG service unavailable")


@router.get("/{case_id}/insights/tasks/{task_id}")
async def insights_task_status(
    case_id: UUID,
    task_id: str,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_case_and_rag_id(case_id, db, current_user)
    url = f"{_agents_base()}/tasks/{task_id}"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError:
        raise HTTPException(status_code=502, detail="RAG insights service returned an error")
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="RAG service unavailable")


# --- Direct agents upload (prefer unified POST /cases/{id}/documents on main backend) ---


@router.post("/{case_id}/insights/upload")
async def insights_upload(
    case_id: UUID,
    file: Annotated[UploadFile, File()],
    file_type: Annotated[str, Form()],
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    if file_type not in VALID_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file_type. Valid: {sorted(VALID_FILE_TYPES)}",
        )
    _, rag_id = await _get_case_and_rag_id(case_id, db, current_user)
    url = f"{_agents_base()}/cases/{rag_id}/upload"
    content = await file.read()
    files = {"file": (file.filename or "upload", content, file.content_type or "application/octet-stream")}
    data = {"file_type": file_type}

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(url, files=files, data=data)
            r.raise_for_status()
            payload = r.json()
            return _rewrite_agent_links(payload, case_id)
    except httpx.HTTPStatusError as exc:
        logger.error("RAG insights upload: %s", exc.response.text)
        raise HTTPException(status_code=502, detail="RAG insights upload failed")
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="RAG service unavailable")


@router.post("/{case_id}/insights/classify-preview")
async def insights_classify_preview(
    case_id: UUID,
    file: Annotated[UploadFile, File()],
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    _, rag_id = await _get_case_and_rag_id(case_id, db, current_user)
    url = f"{_agents_base()}/cases/{rag_id}/classify-preview"
    content = await file.read()
    files = {"file": (file.filename or "upload", content, file.content_type or "application/octet-stream")}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, files=files)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError:
        raise HTTPException(status_code=502, detail="RAG insights classify failed")
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="RAG service unavailable")


@router.get("/{case_id}/insights/memory/{agent_type}")
async def insights_memory(
    case_id: UUID,
    agent_type: str,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    if agent_type not in VALID_FILE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown agent type: {agent_type}")
    _, rag_id = await _get_case_and_rag_id(case_id, db, current_user)
    url = f"{_agents_base()}/cases/{rag_id}/memory/{agent_type}"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError:
        raise HTTPException(status_code=502, detail="RAG insights service returned an error")
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="RAG service unavailable")
