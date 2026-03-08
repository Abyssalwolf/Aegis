from datetime import timedelta
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.core import security
from app.core.config import settings
from app.models.user import User
from app.schemas.token import Token
from app.schemas.user import UserChangePassword
from pydantic import BaseModel

router = APIRouter()

class RefreshRequest(BaseModel):
    refresh_token: str

@router.post("/login", response_model=Token)
async def login_access_token(
    db: AsyncSession = Depends(deps.get_db),
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    result = await db.execute(select(User).filter(User.username == form_data.username))
    user = result.scalars().first()
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return {
        "access_token": security.create_access_token(
            user.id, expires_delta=access_token_expires
        ),
        "refresh_token": security.create_refresh_token(user.id),
        "token_type": "bearer",
    }

@router.post("/refresh", response_model=Token)
async def refresh_token(
    request: RefreshRequest,
    db: AsyncSession = Depends(deps.get_db)
) -> Any:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    try:
        from jose import jwt, JWTError
        from app.schemas.token import TokenPayload
        import uuid
        payload = jwt.decode(request.refresh_token, settings.SECRET_KEY, algorithms=["HS256"])
        token_data = TokenPayload(**payload)
        if not token_data.refresh:
            raise HTTPException(status_code=400, detail="Invalid token type")
        user_uuid = uuid.UUID(token_data.sub)
    except Exception:
        raise credentials_exception
    
    result = await db.execute(select(User).filter(User.id == user_uuid))
    user = result.scalars().first()
    if not user or not user.is_active:
        raise credentials_exception
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return {
        "access_token": security.create_access_token(
            user.id, expires_delta=access_token_expires
        ),
        "refresh_token": security.create_refresh_token(user.id),
        "token_type": "bearer",
    }

@router.post("/change-password")
async def change_password(
    data: UserChangePassword,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
) -> Any:
    # Verify current password
    if not security.verify_password(data.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect current password")
    
    # Update password
    current_user.hashed_password = security.get_password_hash(data.new_password)
    db.add(current_user)
    await db.commit()
    return {"message": "Password updated successfully"}
