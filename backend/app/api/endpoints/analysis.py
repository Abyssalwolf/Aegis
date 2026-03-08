from typing import Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.api.endpoints.cases import check_case_access
from app.models.user import User
from app.models.case import Case

router = APIRouter()

@router.post("/{case_id}/analysis")
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

    return {"message": "Analysis module not enabled yet"}

@router.get("/{case_id}/analysis")
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

    return {"message": "Analysis module not enabled yet"}
