import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Integer, Text, ForeignKey, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.models.base import Base

class Case(Base):
    __tablename__ = "case"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String, index=True)
    description: Mapped[str] = mapped_column(Text)
    created_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id"), index=True)
    required_clearance_level: Mapped[int] = mapped_column(Integer, index=True)
    status: Mapped[str] = mapped_column(String, default="OPEN") # OPEN, UNDER_INVESTIGATION, CLOSED
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    creator = relationship("User", back_populates="cases_created", foreign_keys=[created_by])
    assignments = relationship("CaseAssignment", back_populates="case_link")
    documents = relationship("Document", back_populates="case")
    activities = relationship("ActivityLog", back_populates="case")
    analysis = relationship("CaseAnalysis", back_populates="case")
