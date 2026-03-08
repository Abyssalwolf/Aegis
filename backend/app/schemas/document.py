from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, ConfigDict

class DocumentBase(BaseModel):
    document_type: str
    file_path: str

class DocumentCreate(DocumentBase):
    pass

class DocumentInDBBase(DocumentBase):
    id: UUID
    case_id: UUID
    uploaded_by: UUID
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class Document(DocumentInDBBase):
    pass
