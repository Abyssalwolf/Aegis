from typing import Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import or_, func

from app.api import deps
from app.models.user import User
from app.models.case import Case
from app.models.assignment import CaseAssignment
from app.schemas.user import User as UserSchema, PaginatedUsers
from app.schemas.case import PaginatedCases

router = APIRouter()

@router.get("/me", response_model=UserSchema)
async def read_user_me(
    current_user: User = Depends(deps.get_current_active_officer),
) -> Any:
    return current_user

@router.get("/cases", response_model=PaginatedCases)
async def read_officer_cases(
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_officer),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
) -> Any:
    access_filter = or_(
        Case.created_by == current_user.id,
        CaseAssignment.officer_id == current_user.id,
    )
    count_stmt = (
        select(func.count(func.distinct(Case.id)))
        .select_from(Case)
        .outerjoin(CaseAssignment, Case.id == CaseAssignment.case_id)
        .filter(access_filter)
    )
    total = (await db.execute(count_stmt)).scalar_one()

    list_stmt = (
        select(Case)
        .outerjoin(CaseAssignment, Case.id == CaseAssignment.case_id)
        .filter(access_filter)
        .distinct()
        .order_by(Case.updated_at.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(list_stmt)
    items = result.scalars().all()
    return PaginatedCases(items=items, total=total, skip=skip, limit=limit)

@router.get("/list", response_model=PaginatedUsers)
async def read_officers_list(
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_officer),
    skip: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=1000),
) -> Any:
    # Any officer can see the list of active officers to assign them to cases
    filt = (User.role == "OFFICER") & (User.is_active == True)
    total = (await db.execute(select(func.count()).select_from(User).where(filt))).scalar_one()
    result = await db.execute(select(User).where(filt).order_by(User.username).offset(skip).limit(limit))
    items = result.scalars().all()
    return PaginatedUsers(items=items, total=total, skip=skip, limit=limit)


@router.get("/{officer_id}", response_model=UserSchema)
async def read_officer_by_id(
    officer_id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Get any officer's basic profile — accessible to all authenticated users."""
    result = await db.execute(select(User).filter(User.id == officer_id, User.role == "OFFICER"))
    officer = result.scalars().first()
    if not officer:
        raise HTTPException(status_code=404, detail="Officer not found")
    return officer
