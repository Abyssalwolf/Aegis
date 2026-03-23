from typing import Any, List, Optional
from uuid import UUID
import asyncio
import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Response, status
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.api.endpoints.cases import check_case_access
from app.core.config import settings
from app.core.evidence_category_mapping import map_evidence_category_to_agent_file_type
from app.core.insights_case_id import uuid_to_insights_case_id
from app.models.user import User
from app.models.case import Case
from app.models.document import Document
from app.models.activity import ActivityLog
from app.schemas.document import Document as DocumentSchema, DocumentUploadResponse

logger = logging.getLogger(__name__)
router = APIRouter()

SUPPORTED_CONTENT_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/bmp",
    "image/webp",
}


def _optional_form_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = value.strip()
    return s or None


@router.post("/{case_id}/documents", response_model=DocumentUploadResponse)
async def upload_document(
    case_id: UUID,
    file: UploadFile = File(...),
    display_name: Optional[str] = Form(None),
    evidence_category: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    result = await db.execute(select(Case).filter(Case.id == case_id))
    case = result.scalars().first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if not await check_case_access(db, case, current_user):
        raise HTTPException(status_code=403, detail="Forbidden")

    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Supported: PDF and common image formats.",
        )

    file_bytes = await file.read()
    filename = file.filename or "upload"
    disp = _optional_form_str(display_name)
    ev_cat = _optional_form_str(evidence_category)
    desc = _optional_form_str(description)

    if disp is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="display_name is required.",
        )
    if ev_cat is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="evidence_category is required.",
        )

    agent_file_type = map_evidence_category_to_agent_file_type(ev_cat)
    if not agent_file_type:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "evidence_category must be one of the investigation document types "
                "(FIR, case diary, statement, scene of crime, forensic, seizure, arrest/remand) — not 'Other'."
            ),
        )

    document = Document(
        case_id=case_id,
        uploaded_by=current_user.id,
        document_type=file.content_type or "unknown",
        file_path=f"/rag/{case_id}/{filename}",
        filename=filename,
        display_name=disp,
        evidence_category=ev_cat,
        description=desc,
        ingest_status="processing",
    )
    db.add(document)

    activity = ActivityLog(
        case_id=case_id,
        user_id=current_user.id,
        action="DOCUMENT_UPLOADED",
    )
    db.add(activity)
    await db.commit()
    await db.refresh(document)

    rag_base = settings.RAG_SERVICE_URL.rstrip("/")
    rag_case_int = uuid_to_insights_case_id(case_id)
    ingest_data = {
        "case_id": str(case_id),
        "officer_id": str(current_user.id),
        "display_name": disp,
        "evidence_category": ev_cat,
    }
    if desc is not None:
        ingest_data["description"] = desc

    insights_task_id: Optional[str] = None
    insights_queue_status = "skipped"

    async def post_ingest(client: httpx.AsyncClient) -> httpx.Response:
        return await client.post(
            f"{rag_base}/ingest/file",
            files={"file": (filename, file_bytes, file.content_type)},
            data=ingest_data,
        )

    async def post_agents_queue(client: httpx.AsyncClient) -> httpx.Response:
        return await client.post(
            f"{rag_base}/agents/cases/{rag_case_int}/upload",
            files={"file": (filename, file_bytes, file.content_type)},
            data={"file_type": agent_file_type},
        )

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            ingest_res, agents_res = await asyncio.gather(
                post_ingest(client),
                post_agents_queue(client),
                return_exceptions=True,
            )
    except Exception as exc:
        logger.error("RAG parallel upload setup failed for %s: %s", filename, exc)
        document.ingest_status = "rag_unavailable"
        insights_queue_status = "failed"
        ingest_res, agents_res = None, None

    if isinstance(ingest_res, httpx.Response):
        try:
            ingest_res.raise_for_status()
            rag_data = ingest_res.json()
            document.rag_document_id = rag_data.get("document_id")
            document.ingest_status = rag_data.get("status", "completed")
        except httpx.HTTPStatusError as exc:
            logger.error("RAG ingest HTTP error for %s: %s", filename, exc.response.text)
            document.ingest_status = "failed"
    elif ingest_res is not None:
        logger.error("RAG ingest failed for %s: %s", filename, ingest_res)
        document.ingest_status = "rag_unavailable"

    if isinstance(agents_res, httpx.Response):
        try:
            agents_res.raise_for_status()
            payload = agents_res.json()
            insights_task_id = payload.get("task_id")
            insights_queue_status = "queued" if insights_task_id else "failed"
        except httpx.HTTPStatusError as exc:
            logger.error(
                "RAG agents queue HTTP error for %s: %s",
                filename,
                exc.response.text,
            )
            insights_queue_status = "failed"
    elif agents_res is not None:
        logger.error("RAG agents queue failed for %s: %s", filename, agents_res)
        insights_queue_status = "failed"

    await db.commit()
    await db.refresh(document)

    base = DocumentUploadResponse.model_validate(document, from_attributes=True)
    return base.model_copy(
        update={
            "insights_task_id": insights_task_id,
            "insights_queue_status": insights_queue_status,
        }
    )


@router.get("/{case_id}/documents", response_model=List[DocumentSchema])
async def get_documents(
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

    result = await db.execute(select(Document).filter(Document.case_id == case_id))
    return result.scalars().all()


@router.delete("/{case_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_case_document(
    case_id: UUID,
    document_id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Response:
    result = await db.execute(select(Case).filter(Case.id == case_id))
    case = result.scalars().first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if not await check_case_access(db, case, current_user):
        raise HTTPException(status_code=403, detail="Forbidden")

    result = await db.execute(
        select(Document).filter(Document.id == document_id, Document.case_id == case_id)
    )
    document = result.scalars().first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document.rag_document_id:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.delete(
                    f"{settings.RAG_SERVICE_URL}/ingest/documents/{document.rag_document_id}",
                )
                if r.status_code not in (200, 204, 404):
                    r.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "RAG delete failed for rag_document_id=%s: %s",
                document.rag_document_id,
                exc,
            )

    await db.execute(delete(Document).where(Document.id == document_id))
    activity = ActivityLog(
        case_id=case_id,
        user_id=current_user.id,
        action="DOCUMENT_DELETED",
    )
    db.add(activity)
    await db.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)
