from typing import Any, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.core.security import get_password_hash
from app.models.user import User
from app.schemas.user import User as UserSchema, UserCreate, UserUpdate

router = APIRouter()

@router.get("/officers", response_model=List[UserSchema])
async def read_officers(
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_admin),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    result = await db.execute(select(User).filter(User.role == "OFFICER").offset(skip).limit(limit))
    return result.scalars().all()

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
    await db.delete(user)
    await db.commit()
    return user
