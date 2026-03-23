from typing import Optional, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, ConfigDict

class UserBase(BaseModel):
    username: str
    role: str
    rank: Optional[str] = None
    clearance_level: Optional[int] = None
    badge_number: Optional[str] = None
    station_name: Optional[str] = None
    is_active: bool = True

class UserCreate(UserBase):
    password: str
    rank: str  # Rank is required for Officer creation
    clearance_level: int  # Required for Officer creation
    badge_number: Optional[str] = None
    station_name: str  # Required for Officer creation

class UserUpdate(BaseModel):
    role: Optional[str] = None
    rank: Optional[str] = None
    clearance_level: Optional[int] = None
    badge_number: Optional[str] = None
    station_name: Optional[str] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None

class UserChangePassword(BaseModel):
    current_password: str
    new_password: str

class UserInDBBase(UserBase):
    id: UUID
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class User(UserInDBBase):
    pass


class PaginatedUsers(BaseModel):
    items: List[User]
    total: int
    skip: int
    limit: int
