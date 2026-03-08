"""
SQLite document store.
Tracks every ingested document with its status, metadata, and chunk count.
Provides an audit trail of what has been ingested and when.
"""

import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Optional

from config.settings import settings
from core.documents.models import DocumentRecord, DocumentMetadata, DocumentStatus

logger = logging.getLogger(__name__)


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id     TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    source_path     TEXT NOT NULL,
    file_type       TEXT NOT NULL,
    case_id         TEXT,
    officer_id      TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    chunk_count     INTEGER DEFAULT 0,
    page_count      INTEGER,
    error_message   TEXT,
    extra_metadata  TEXT,           -- JSON blob
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_case_id ON documents(case_id);
CREATE INDEX IF NOT EXISTS idx_documents_status  ON documents(status);
"""


class DocumentStore:

    def __init__(self, db_path: Path = settings.document_store_path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(CREATE_TABLE_SQL)
        logger.info(f"Document store initialized at: {self.db_path}")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def create(self, record: DocumentRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id, filename, source_path, file_type,
                    case_id, officer_id, status, chunk_count, page_count,
                    error_message, extra_metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.document_id,
                    record.metadata.filename,
                    record.metadata.source_path,
                    record.metadata.file_type,
                    record.metadata.case_id,
                    record.metadata.officer_id,
                    record.status.value,
                    record.chunk_count,
                    record.metadata.page_count,
                    record.error_message,
                    json.dumps(record.metadata.extra),
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )
        logger.debug(f"Created document record: {record.document_id}")

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        chunk_count: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE documents
                SET status = ?, chunk_count = ?, error_message = ?, updated_at = ?
                WHERE document_id = ?
                """,
                (
                    status.value,
                    chunk_count,
                    error_message,
                    datetime.utcnow().isoformat(),
                    document_id,
                ),
            )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, document_id: str) -> Optional[DocumentRecord]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def exists_by_path(self, source_path: str) -> bool:
        """Check if a file has already been ingested (by path)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE source_path = ? AND status = 'completed'",
                (source_path,),
            ).fetchone()
        return row is not None

    def list_by_case(self, case_id: str) -> list[DocumentRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE case_id = ? ORDER BY created_at DESC",
                (case_id,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_all(self, limit: int = 100, offset: int = 0) -> list[DocumentRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row_to_record(self, row: sqlite3.Row) -> DocumentRecord:
        extra = json.loads(row["extra_metadata"] or "{}")
        metadata = DocumentMetadata(
            source_path=row["source_path"],
            filename=row["filename"],
            file_type=row["file_type"],
            case_id=row["case_id"],
            officer_id=row["officer_id"],
            page_count=row["page_count"],
            extra=extra,
        )
        return DocumentRecord(
            document_id=row["document_id"],
            metadata=metadata,
            status=DocumentStatus(row["status"]),
            chunk_count=row["chunk_count"] or 0,
            error_message=row["error_message"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
