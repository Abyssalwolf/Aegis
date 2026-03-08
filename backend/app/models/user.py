import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Integer, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.models.base import Base

class User(Base):
    __tablename__ = "user"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String) # ADMIN or OFFICER
    rank: Mapped[str] = mapped_column(String, nullable=True)
    clearance_level: Mapped[int] = mapped_column(Integer, index=True, nullable=True)
    badge_number: Mapped[str] = mapped_column(String, nullable=True)
    station_name: Mapped[str] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    cases_created = relationship("Case", back_populates="creator", foreign_keys="Case.created_by")
    assigned_cases = relationship("CaseAssignment", back_populates="officer")
    documents_uploaded = relationship("Document", back_populates="uploader")
