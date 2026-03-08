from typing import Any, List
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import or_

from app.api import deps
from app.models.user import User
from app.models.case import Case
from app.models.assignment import CaseAssignment
from app.schemas.user import User as UserSchema
from app.schemas.case import Case as CaseSchema

router = APIRouter()

@router.get("/me", response_model=UserSchema)
async def read_user_me(
    current_user: User = Depends(deps.get_current_active_officer),
) -> Any:
    return current_user

@router.get("/cases", response_model=List[CaseSchema])
async def read_officer_cases(
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_officer),
) -> Any:
    query = select(Case).outerjoin(
        CaseAssignment, Case.id == CaseAssignment.case_id
    ).filter(
        or_(
            Case.created_by == current_user.id,
            CaseAssignment.officer_id == current_user.id
        )
    ).distinct()
    
    result = await db.execute(query)
    return result.scalars().all()

@router.get("/list", response_model=List[UserSchema])
async def read_officers_list(
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_officer),
) -> Any:
    # Any officer can see the list of active officers to assign them to cases
    query = select(User).filter(User.role == "OFFICER", User.is_active == True)
    result = await db.execute(query)
    return result.scalars().all()
