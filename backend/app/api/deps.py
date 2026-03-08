from typing import AsyncGenerator
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.models.user import User
from app.schemas.token import TokenPayload
import uuid

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

async def get_current_user(
    db: AsyncSession = Depends(get_db), token: str = Depends(oauth2_scheme)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        token_data = TokenPayload(**payload)
        if token_data.refresh:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    try:
        user_uuid = uuid.UUID(token_data.sub)
    except Exception:
        raise credentials_exception

    result = await db.execute(select(User).filter(User.id == user_uuid))
    user = result.scalars().first()
    if not user:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user

async def get_current_active_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    if current_user.role != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="The user doesn't have enough privileges"
        )
    return current_user

async def get_current_active_officer(
    current_user: User = Depends(get_current_user),
) -> User:
    if current_user.role != "OFFICER":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Only officers can perform this action"
        )
    return current_user
