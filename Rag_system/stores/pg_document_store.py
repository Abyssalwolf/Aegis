"""
PostgreSQL document store.
Drop-in replacement for document_store.py (SQLite) backed by Neon.tech PostgreSQL.
Uses psycopg2 (sync) to match the synchronous interface of the original SQLite store.
Auto-creates the rag_documents table on first init.
"""

import json
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import Optional

import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection as PgConnection

from core.documents.models import DocumentRecord, DocumentMetadata, DocumentStatus

logger = logging.getLogger(__name__)


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS rag_documents (
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
    extra_metadata  TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rag_documents_case_id ON rag_documents(case_id);
CREATE INDEX IF NOT EXISTS idx_rag_documents_status  ON rag_documents(status);
"""


class PgDocumentStore:
    """Synchronous PostgreSQL-backed document store for RAG ingestion tracking."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._init_db()

    @contextmanager
    def _conn(self):
        conn: PgConnection = psycopg2.connect(self.database_url)
        conn.autocommit = False
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
            with conn.cursor() as cur:
                for statement in CREATE_TABLE_SQL.strip().split(";"):
                    stmt = statement.strip()
                    if stmt:
                        cur.execute(stmt)
        logger.info("PgDocumentStore initialized (rag_documents table ensured).")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def create(self, record: DocumentRecord) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_documents (
                        document_id, filename, source_path, file_type,
                        case_id, officer_id, status, chunk_count, page_count,
                        error_message, extra_metadata, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id) DO NOTHING
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
        logger.debug(f"Created RAG document record: {record.document_id}")

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        chunk_count: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE rag_documents
                    SET status = %s, chunk_count = %s, error_message = %s, updated_at = %s
                    WHERE document_id = %s
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
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM rag_documents WHERE document_id = %s",
                    (document_id,),
                )
                row = cur.fetchone()
        return self._row_to_record(dict(row)) if row else None

    def exists_by_path(self, source_path: str) -> bool:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM rag_documents WHERE source_path = %s AND status = 'completed'",
                    (source_path,),
                )
                return cur.fetchone() is not None

    def list_by_case(self, case_id: str) -> list[DocumentRecord]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM rag_documents WHERE case_id = %s ORDER BY created_at DESC",
                    (case_id,),
                )
                rows = cur.fetchall()
        return [self._row_to_record(dict(r)) for r in rows]

    def list_all(self, limit: int = 100, offset: int = 0) -> list[DocumentRecord]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM rag_documents ORDER BY created_at DESC LIMIT %s OFFSET %s",
                    (limit, offset),
                )
                rows = cur.fetchall()
        return [self._row_to_record(dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row_to_record(self, row: dict) -> DocumentRecord:
        extra = json.loads(row.get("extra_metadata") or "{}")
        metadata = DocumentMetadata(
            source_path=row["source_path"],
            filename=row["filename"],
            file_type=row["file_type"],
            case_id=row.get("case_id"),
            officer_id=row.get("officer_id"),
            page_count=row.get("page_count"),
            extra=extra,
        )
        return DocumentRecord(
            document_id=row["document_id"],
            metadata=metadata,
            status=DocumentStatus(row["status"]),
            chunk_count=row.get("chunk_count") or 0,
            error_message=row.get("error_message"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
