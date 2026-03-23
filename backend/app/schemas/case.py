from typing import Optional, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, ConfigDict

class CaseBase(BaseModel):
    title: str
    description: str
    required_clearance_level: int
    status: Optional[str] = "OPEN"

class CaseCreate(CaseBase):
    assigned_officer_ids: Optional[List[UUID]] = []

class CaseUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    required_clearance_level: Optional[int] = None
    status: Optional[str] = None

class CaseInDBBase(CaseBase):
    id: UUID
    created_by: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class Case(CaseInDBBase):
    pass

class CaseTransfer(BaseModel):
    new_owner_id: UUID


class PaginatedCases(BaseModel):
    """Paginated case list for officer/admin dashboards."""

    items: List[Case]
    total: int
    skip: int
    limit: int
