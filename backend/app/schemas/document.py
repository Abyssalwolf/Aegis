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
    filename: Optional[str] = None
    display_name: Optional[str] = None
    evidence_category: Optional[str] = None
    description: Optional[str] = None
    rag_document_id: Optional[str] = None
    ingest_status: str = "pending"
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class Document(DocumentInDBBase):
    pass
