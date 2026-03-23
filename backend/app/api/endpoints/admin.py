from typing import Any, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete, func

from app.api import deps
from app.core.security import get_password_hash
from app.models.user import User
from app.models.case import Case
from app.models.assignment import CaseAssignment
from app.models.activity import ActivityLog
from app.models.document import Document
from app.schemas.user import User as UserSchema, UserCreate, UserUpdate, PaginatedUsers
from app.schemas.case import Case as CaseSchema, PaginatedCases
from app.schemas.assignment import CaseAssignmentCreate

router = APIRouter()

@router.get("/officers", response_model=PaginatedUsers)
async def read_officers(
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_admin),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
) -> Any:
    filt = User.role == "OFFICER"
    total = (await db.execute(select(func.count()).select_from(User).where(filt))).scalar_one()
    result = await db.execute(
        select(User).where(filt).order_by(User.username).offset(skip).limit(limit)
    )
    items = result.scalars().all()
    return PaginatedUsers(items=items, total=total, skip=skip, limit=limit)

@router.post("/officers", response_model=UserSchema)
async def create_officer(
    *,
    db: AsyncSession = Depends(deps.get_db),
    user_in: UserCreate,
    current_user: User = Depends(deps.get_current_active_admin),
) -> Any:
    result = await db.execute(select(User).filter(User.username == user_in.username))
    if result.scalars().first():
        raise HTTPException(
            status_code=400,
            detail="The user with this username already exists in the system.",
        )
    user = User(
        username=user_in.username,
        hashed_password=get_password_hash(user_in.password),
        role=user_in.role,
        rank=user_in.rank,
        clearance_level=user_in.clearance_level,
        badge_number=user_in.badge_number,
        station_name=user_in.station_name,
        is_active=user_in.is_active,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

@router.patch("/officers/{user_id}", response_model=UserSchema)
async def update_officer(
    *,
    db: AsyncSession = Depends(deps.get_db),
    user_id: UUID,
    user_in: UserUpdate,
    current_user: User = Depends(deps.get_current_active_admin),
) -> Any:
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    update_data = user_in.model_dump(exclude_unset=True)
    if "password" in update_data:
        hashed_password = get_password_hash(update_data.pop("password"))
        update_data["hashed_password"] = hashed_password
        
    for field, value in update_data.items():
        setattr(user, field, value)
        
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

@router.patch("/officers/{user_id}/status", response_model=UserSchema)
async def update_officer_status(
    *,
    db: AsyncSession = Depends(deps.get_db),
    user_id: UUID,
    is_active: bool,
    current_user: User = Depends(deps.get_current_active_admin),
) -> Any:
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.is_active = is_active
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

@router.delete("/officers/{user_id}", response_model=UserSchema)
async def delete_officer(
    *,
    db: AsyncSession = Depends(deps.get_db),
    user_id: UUID,
    current_user: User = Depends(deps.get_current_active_admin),
) -> Any:
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Block deletion if the officer owns any cases — admin must transfer or delete those cases first
    owned_cases = await db.execute(select(Case).filter(Case.created_by == user_id).limit(1))
    if owned_cases.scalars().first():
        raise HTTPException(
            status_code=400,
            detail="Cannot delete officer: they are the creator of one or more cases. "
                   "Transfer or delete those cases first."
        )

    # Clean up all FK references before deleting the user
    await db.execute(delete(CaseAssignment).where(CaseAssignment.officer_id == user_id))
    await db.execute(delete(ActivityLog).where(ActivityLog.user_id == user_id))
    await db.execute(delete(Document).where(Document.uploaded_by == user_id))

    await db.delete(user)
    await db.commit()
    return user


# ── Admin Case Management ─────────────────────────────────────────────────────

@router.get("/cases", response_model=PaginatedCases)
async def admin_list_cases(
    officer_id: Optional[UUID] = Query(None),
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_admin),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
) -> Any:
    """List all cases, optionally filtered by the officer who created them."""
    count_stmt = select(func.count()).select_from(Case)
    list_stmt = select(Case)
    if officer_id:
        count_stmt = count_stmt.where(Case.created_by == officer_id)
        list_stmt = list_stmt.where(Case.created_by == officer_id)
    total = (await db.execute(count_stmt)).scalar_one()
    result = await db.execute(list_stmt.order_by(Case.created_at.desc()).offset(skip).limit(limit))
    items = result.scalars().all()
    return PaginatedCases(items=items, total=total, skip=skip, limit=limit)


@router.delete("/cases/{case_id}")
async def admin_delete_case(
    case_id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_admin),
) -> Any:
    """Admin can delete any case regardless of ownership."""
    result = await db.execute(select(Case).filter(Case.id == case_id))
    case = result.scalars().first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    await db.execute(delete(CaseAssignment).where(CaseAssignment.case_id == case_id))
    await db.execute(delete(ActivityLog).where(ActivityLog.case_id == case_id))
    await db.execute(delete(Document).where(Document.case_id == case_id))
    await db.delete(case)
    await db.commit()
    return {"message": "Case deleted successfully"}


@router.post("/cases/{case_id}/officers", response_model=UserSchema)
async def admin_assign_officer(
    case_id: UUID,
    assignment_in: CaseAssignmentCreate,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_admin),
) -> Any:
    """Admin can assign any active officer whose clearance >= case required level."""
    result = await db.execute(select(Case).filter(Case.id == case_id))
    case = result.scalars().first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    officer_res = await db.execute(select(User).filter(User.id == assignment_in.officer_id))
    officer = officer_res.scalars().first()
    if not officer or officer.role != "OFFICER" or not officer.is_active:
        raise HTTPException(status_code=400, detail="Invalid or inactive officer.")

    if (officer.clearance_level or 0) < case.required_clearance_level:
        raise HTTPException(
            status_code=400,
            detail=f"Officer clearance level ({officer.clearance_level}) is below the case requirement ({case.required_clearance_level})."
        )

    existing = await db.execute(
        select(CaseAssignment).filter(
            CaseAssignment.case_id == case_id,
            CaseAssignment.officer_id == officer.id
        )
    )
    if existing.scalars().first():
        raise HTTPException(status_code=400, detail="Officer is already assigned to this case.")

    db.add(CaseAssignment(case_id=case_id, officer_id=officer.id))
    db.add(ActivityLog(case_id=case_id, user_id=current_user.id, action=f"ADMIN_ASSIGNED_OFFICER:{officer.id}"))
    await db.commit()
    return officer


@router.delete("/cases/{case_id}/officers/{officer_id}")
async def admin_remove_officer(
    case_id: UUID,
    officer_id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_admin),
) -> Any:
    """Admin can remove any assigned officer except the case creator."""
    result = await db.execute(select(Case).filter(Case.id == case_id))
    case = result.scalars().first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if officer_id == case.created_by:
        raise HTTPException(status_code=400, detail="Cannot remove the original creator of the case.")

    assign_res = await db.execute(
        select(CaseAssignment).filter(
            CaseAssignment.case_id == case_id,
            CaseAssignment.officer_id == officer_id
        )
    )
    assignment = assign_res.scalars().first()
    if not assignment:
        raise HTTPException(status_code=404, detail="Officer is not assigned to this case.")

    await db.delete(assignment)
    db.add(ActivityLog(case_id=case_id, user_id=current_user.id, action=f"ADMIN_REMOVED_OFFICER:{officer_id}"))
    await db.commit()
    return {"message": "Officer removed successfully"}
