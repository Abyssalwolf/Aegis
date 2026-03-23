from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class CaseAnalysisOut(BaseModel):
    id: UUID
    case_id: UUID
    analysis_type: str
    result_text: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
