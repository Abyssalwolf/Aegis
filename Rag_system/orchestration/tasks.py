import sys
import os

# Ensure the Rag_system project root is on sys.path so packages like
# `ingestion`, `core`, `agents` are importable by Celery workers.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import logging
from pathlib import Path
from typing import Any

# All project imports at module level — resolved once when Celery loads this
# module, at which point sys.path above is already applied.
from ingestion.extractor import extract_text
from core.documents.classifier import classify_document
from core.retrieval.agent_retriever import ingest_document
from agents.supervisor import SupervisorAgent
from orchestration.celery_app import celery_app
from orchestration.blackboard import set_case_status, post_insight
from orchestration.graph.graph import investigation_graph
from orchestration.graph.state import InvestigationState

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def process_document(self, case_id: int, file_path: str,
                     stated_file_type: str | None = None) -> dict[str, Any]:
    """
    Full pipeline for an uploaded case document:
      1. Extract text (PDF / DOCX / TXT / image)
      2. Classify document type
      3. Ingest into RAG store
      4. Run LangGraph → specialist agent → supervisor
      5. Post findings to blackboard
    """
    try:
        set_case_status(case_id, "running")

        file_content = extract_text(file_path)
        if not file_content.strip():
            return {"case_id": case_id, "status": "error",
                    "error": "Could not extract text from file."}

        classification = classify_document(
            text=file_content,
            stated_type=stated_file_type,
        )
        file_type = classification["file_type"]
        logger.info(f"[{case_id}] Classified as '{file_type}' "
                    f"(conf: {classification['confidence']})")

        ingest_document(
            case_id=str(case_id),
            doc_id=f"{case_id}_{file_type}_{Path(file_path).stem}",
            text=file_content,
            metadata={"file_type": file_type, "file_path": file_path},
        )

        final_state = investigation_graph.invoke(InvestigationState(
            case_id=str(case_id),
            file_type=file_type,
            file_path=file_path,
            file_content=file_content,
        ))

        set_case_status(case_id, "completed")
        return {
            "case_id": case_id,
            "file_type": file_type,
            "status": final_state.get("final_status", "analysed"),
            "supervisor_report": final_state.get("supervisor_report", ""),
            "cross_inconsistencies": final_state.get("cross_inconsistencies", []),
        }

    except Exception as exc:
        logger.error(f"[{case_id}] Failed: {exc}", exc_info=True)
        set_case_status(case_id, "error")
        raise self.retry(exc=exc, countdown=15)


@celery_app.task
def run_supervisor(case_id: int) -> dict[str, Any]:
    """
    Re-run the supervisor over the current blackboard.
    Call this after all documents for a case have been uploaded
    or to refresh the consolidated report on demand.
    """
    result = SupervisorAgent()(InvestigationState(case_id=str(case_id)))

    for issue in result.get("cross_inconsistencies", []):
        post_insight(case_id, "supervisor", issue, confidence=0.95)

    set_case_status(case_id, "completed")
    return result


@celery_app.task
def classify_only(file_path: str, stated_type: str | None = None) -> dict[str, Any]:
    """Classify a file without running the full pipeline — for UI preview."""
    return classify_document(text=extract_text(file_path), stated_type=stated_type)