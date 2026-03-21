from typing import Any, List, Optional
from uuid import UUID
import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.api.endpoints.cases import check_case_access
from app.core.config import settings
from app.models.user import User
from app.models.case import Case
from app.models.document import Document
from app.models.activity import ActivityLog
from app.schemas.document import Document as DocumentSchema

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


@router.post("/{case_id}/documents", response_model=DocumentSchema)
async def upload_document(
    case_id: UUID,
    file: UploadFile = File(...),
    display_name: Optional[str] = Form(default=None),
    evidence_category: Optional[str] = Form(default=None),
    description: Optional[str] = Form(default=None),
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

    document = Document(
        case_id=case_id,
        uploaded_by=current_user.id,
        document_type=file.content_type or "unknown",
        file_path=f"/rag/{case_id}/{filename}",
        filename=filename,
        display_name=display_name or filename,
        evidence_category=evidence_category,
        description=description,
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

    rag_form_data: dict[str, str] = {
        "case_id": str(case_id),
        "officer_id": str(current_user.id),
    }
    if display_name:
        rag_form_data["display_name"] = display_name
    if evidence_category:
        rag_form_data["evidence_category"] = evidence_category

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{settings.RAG_SERVICE_URL}/ingest/file",
                files={"file": (filename, file_bytes, file.content_type)},
                data=rag_form_data,
            )
            response.raise_for_status()
            rag_data = response.json()

            document.rag_document_id = rag_data.get("document_id")
            document.ingest_status = rag_data.get("status", "completed")
    except httpx.HTTPStatusError as exc:
        logger.error(f"RAG ingest HTTP error for {filename}: {exc.response.text}")
        document.ingest_status = "failed"
    except httpx.RequestError as exc:
        logger.error(f"RAG service unreachable when ingesting {filename}: {exc}")
        document.ingest_status = "rag_unavailable"

    await db.commit()
    await db.refresh(document)
    return document


@router.delete("/{case_id}/documents/{document_id}")
async def delete_document(
    case_id: UUID,
    document_id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    result = await db.execute(select(Case).filter(Case.id == case_id))
    case = result.scalars().first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if not await check_case_access(db, case, current_user):
        raise HTTPException(status_code=403, detail="Forbidden")

    if current_user.role == "OFFICER":
        user_clearance = current_user.clearance_level or 0
        if user_clearance <= case.required_clearance_level:
            raise HTTPException(
                status_code=403,
                detail="Forbidden: Your clearance level must be higher than the case's required level to delete documents.",
            )

    result = await db.execute(
        select(Document).filter(Document.id == document_id, Document.case_id == case_id)
    )
    document = result.scalars().first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document.rag_document_id:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.delete(
                    f"{settings.RAG_SERVICE_URL}/ingest/documents/{document.rag_document_id}"
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(f"RAG delete HTTP error for {document.rag_document_id}: {exc.response.text}")
        except httpx.RequestError as exc:
            logger.error(f"RAG service unreachable when deleting {document.rag_document_id}: {exc}")

    activity = ActivityLog(
        case_id=case_id,
        user_id=current_user.id,
        action="DOCUMENT_DELETED",
    )
    db.add(activity)

    await db.delete(document)
    await db.commit()

    return {"message": "Document deleted successfully"}


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
