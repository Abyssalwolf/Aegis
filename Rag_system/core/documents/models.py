from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime
import uuid


class ChunkType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DocumentMetadata:
    """Metadata extracted from a source document."""
    source_path: str
    filename: str
    file_type: str                          # "pdf" or "image"
    ingested_at: datetime = field(default_factory=datetime.utcnow)
    case_id: Optional[str] = None           # Link to a police case
    officer_id: Optional[str] = None        # Who ingested this
    page_count: Optional[int] = None
    extra: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A single retrievable unit of content."""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""                   # FK to DocumentRecord
    chunk_type: ChunkType = ChunkType.TEXT
    text: str = ""                          # Text content or image caption
    page_number: Optional[int] = None
    chunk_index: int = 0                    # Position within document
    token_count: int = 0
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)
    # For parent-child: store the broader parent section text for context packing
    parent_text: Optional[str] = None


@dataclass
class DocumentRecord:
    """Tracks an ingested document in the document store."""
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: DocumentMetadata = field(default_factory=lambda: DocumentMetadata("", "", ""))
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval with its score."""
    chunk: Chunk
    score: float
    retrieval_method: str   # "dense", "sparse", "hybrid", "reranked"
