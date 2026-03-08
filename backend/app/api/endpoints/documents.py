from typing import Any, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.api.endpoints.cases import check_case_access
from app.models.user import User
from app.models.case import Case
from app.models.document import Document
from app.models.activity import ActivityLog
from app.schemas.document import Document as DocumentSchema

router = APIRouter()

@router.post("/{case_id}/documents", response_model=DocumentSchema)
async def upload_document(
    case_id: UUID,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    result = await db.execute(select(Case).filter(Case.id == case_id))
    case = result.scalars().first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
        
    if not await check_case_access(db, case, current_user):
        raise HTTPException(status_code=403, detail="Forbidden")

    # Mock file saving
    file_path = f"/storage/cases/{case_id}/{file.filename}"
    
    document = Document(
        case_id=case_id,
        uploaded_by=current_user.id,
        document_type=file.content_type or "unknown",
        file_path=file_path,
    )
    db.add(document)
    
    activity = ActivityLog(
        case_id=case_id,
        user_id=current_user.id,
        action="DOCUMENT_UPLOADED"
    )
    db.add(activity)
    
    await db.commit()
    await db.refresh(document)
    return document

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
