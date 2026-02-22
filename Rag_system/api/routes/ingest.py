"""
Ingest routes.
POST /ingest/file     — upload a single PDF or image
POST /ingest/batch    — upload multiple files
GET  /ingest/status/{document_id}
GET  /ingest/documents
"""

import tempfile
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ingestion.pipeline import IngestionPipeline
from stores.document_store import DocumentStore
from config.settings import settings

router = APIRouter()

# Shared instances — loaded once per process
_pipeline: Optional[IngestionPipeline] = None
_doc_store: Optional[DocumentStore] = None


def get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


def get_doc_store() -> DocumentStore:
    global _doc_store
    if _doc_store is None:
        _doc_store = DocumentStore()
    return _doc_store


SUPPORTED_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/bmp",
    "image/webp",
}


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: int
    error_message: Optional[str] = None


@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    case_id: Optional[str] = Form(default=None),
    officer_id: Optional[str] = Form(default=None),
):
    """Upload and ingest a single PDF or image file."""
    # Validate file type
    if file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Supported: PDF and common image formats.",
        )

    # Check size
    content = await file.read()
    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB.",
        )

    # Write to temp file (loaders expect file paths)
    suffix = Path(file.filename or "upload").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pipeline = get_pipeline()
        record = pipeline.ingest_file(
            file_path=tmp_path,
            case_id=case_id,
            officer_id=officer_id,
        )
        return IngestResponse(
            document_id=record.document_id,
            filename=file.filename or "",
            status=record.status.value,
            chunk_count=record.chunk_count,
            error_message=record.error_message,
        )
    finally:
        os.unlink(tmp_path)


@router.post("/batch")
async def ingest_batch(
    files: list[UploadFile] = File(...),
    case_id: Optional[str] = Form(default=None),
    officer_id: Optional[str] = Form(default=None),
):
    """Upload and ingest multiple files in one request."""
    results = []
    for file in files:
        content = await file.read()
        suffix = Path(file.filename or "upload").suffix or ".pdf"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            pipeline = get_pipeline()
            record = pipeline.ingest_file(
                file_path=tmp_path,
                case_id=case_id,
                officer_id=officer_id,
            )
            results.append({
                "filename": file.filename,
                "document_id": record.document_id,
                "status": record.status.value,
                "chunk_count": record.chunk_count,
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e),
            })
        finally:
            os.unlink(tmp_path)

    return {"results": results, "total": len(results)}


@router.get("/status/{document_id}")
def get_status(document_id: str):
    """Get ingestion status for a specific document."""
    doc_store = get_doc_store()
    record = doc_store.get(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {
        "document_id": record.document_id,
        "filename": record.metadata.filename,
        "status": record.status.value,
        "chunk_count": record.chunk_count,
        "case_id": record.metadata.case_id,
        "created_at": record.created_at.isoformat(),
        "error_message": record.error_message,
    }


@router.get("/documents")
def list_documents(case_id: Optional[str] = None, limit: int = 50, offset: int = 0):
    """List all ingested documents, optionally filtered by case_id."""
    doc_store = get_doc_store()
    if case_id:
        records = doc_store.list_by_case(case_id)
    else:
        records = doc_store.list_all(limit=limit, offset=offset)

    return {
        "documents": [
            {
                "document_id": r.document_id,
                "filename": r.metadata.filename,
                "status": r.status.value,
                "chunk_count": r.chunk_count,
                "case_id": r.metadata.case_id,
                "file_type": r.metadata.file_type,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ],
        "total": len(records),
    }
