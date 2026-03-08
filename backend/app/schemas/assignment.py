from pydantic import BaseModel
from uuid import UUID

class CaseAssignmentCreate(BaseModel):
    officer_id: UUID
