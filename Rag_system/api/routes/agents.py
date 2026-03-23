"""
api/routes/agents.py — Document agent endpoints.
Registered in app.py as: app.include_router(agents_router, prefix="/agents", tags=["Agents"])
"""

from __future__ import annotations
import asyncio
import threading
import uuid
from pathlib import Path
from typing import Annotated, AsyncGenerator

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from config.settings import settings
from orchestration.celery_app import celery_app
from orchestration.blackboard import (
    read_all,
    format_brief,
    subscribe_to_case,
    get_case_status,
)
from core.documents.manager import read_full_memory

router = APIRouter()

VALID_FILE_TYPES = {
    "fir", "case_diary", "statement", "scene_of_crime",
    "forensic", "seizure", "arrest_remand",
}


# ── Upload & trigger pipeline ─────────────────────────────────────────────────

@router.post("/cases/{case_id}/upload")
async def upload_document(
    case_id: int,
    file: Annotated[UploadFile, File()],
    file_type: Annotated[str, Form()],
):
    if file_type not in VALID_FILE_TYPES:
        raise HTTPException(400, f"Invalid file_type. Valid: {sorted(VALID_FILE_TYPES)}")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_id = uuid.uuid4().hex[:8]
    save_path = upload_dir / f"{case_id}_{file_type}_{file_id}_{file.filename}"

    # Sync disk write after async read — avoids aiofiles + default ThreadPoolExecutor,
    # which can raise "Executor shutdown has been called" if uvicorn stops mid-request.
    save_path.write_bytes(await file.read())

    from orchestration.tasks import process_document
    task = process_document.delay(
        case_id=case_id,
        file_path=str(save_path),
        stated_file_type=file_type,
    )

    return {
        "task_id": task.id,
        "case_id": case_id,
        "file_type": file_type,
        "filename": file.filename,
        "status": "queued",
        "poll_url": f"/agents/tasks/{task.id}",
        "blackboard_url": f"/agents/cases/{case_id}/blackboard",
        "stream_url": f"/agents/cases/{case_id}/stream",
    }


# ── Classify preview ──────────────────────────────────────────────────────────

@router.post("/cases/{case_id}/classify-preview")
async def classify_preview(case_id: int, file: Annotated[UploadFile, File()]):
    """Classify without running full analysis — instant UI feedback."""
    content = await file.read()
    tmp = Path(settings.upload_dir) / f"tmp_{uuid.uuid4().hex}"
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

    tmp.write_bytes(content)

    try:
        from ingestion.extractor import extract_text
        from core.documents.classifier import classify_document
        result = classify_document(text=extract_text(str(tmp)), use_llm_fallback=False)
    finally:
        tmp.unlink(missing_ok=True)

    return {"case_id": case_id, "filename": file.filename, **result}


# ── Task status ───────────────────────────────────────────────────────────────

@router.get("/tasks/{task_id}")
def task_status(task_id: str):
    result = celery_app.AsyncResult(task_id)
    resp = {"task_id": task_id, "status": result.status}
    if result.ready():
        resp["result"] = result.get() if result.successful() else {"error": str(result.result)}
    return resp


# ── Blackboard ────────────────────────────────────────────────────────────────

@router.get("/cases/{case_id}/blackboard")
def get_blackboard(case_id: int):
    """Returns all messages, anomalies, findings and insights for a case."""
    data = read_all(case_id)
    data["case_id"] = case_id
    data["status"] = get_case_status(case_id)
    return data


@router.get("/cases/{case_id}/blackboard/brief")
def get_brief(case_id: int):
    """Returns the full markdown brief (what the supervisor reads)."""
    return {"case_id": case_id, "brief": format_brief(case_id)}


# ── SSE live stream ───────────────────────────────────────────────────────────

@router.get("/cases/{case_id}/stream")
async def stream_blackboard(case_id: int):
    """Server-Sent Events — streams live blackboard updates to the dashboard."""
    async def gen() -> AsyncGenerator[str, None]:
        # send existing findings first
        for item in read_all(case_id).get("findings", []):
            import json
            yield f"data: {json.dumps(item)}\n\n"

        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _sub():
            for msg in subscribe_to_case(case_id):
                loop.call_soon_threadsafe(q.put_nowait, msg)

        threading.Thread(target=_sub, daemon=True).start()

        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=30)
                import json
                yield f"data: {json.dumps(msg)}\n\n"
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── Supervisor report ─────────────────────────────────────────────────────────

@router.post("/cases/{case_id}/report")
def generate_report(case_id: int):
    """Trigger supervisor re-run after all documents are uploaded."""
    from orchestration.tasks import run_supervisor
    task = run_supervisor.delay(case_id)
    return {"task_id": task.id, "poll_url": f"/agents/tasks/{task.id}"}


# ── Agent memory ──────────────────────────────────────────────────────────────

@router.get("/cases/{case_id}/memory/{agent_type}")
def get_memory(case_id: int, agent_type: str):
    if agent_type not in VALID_FILE_TYPES:
        raise HTTPException(400, f"Unknown agent type: {agent_type}")
    return {
        "case_id": case_id,
        "agent_type": agent_type,
        "memory": read_full_memory(str(case_id), agent_type),
    }