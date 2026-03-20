import sys
import os

# Ensure the project root (Rag_system/) is on sys.path so that sibling packages
# like `ingestion`, `core`, `agents` etc. are importable by Celery workers.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from celery import Celery

celery_app = Celery(
    "aegis",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
    include=["orchestration.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
)