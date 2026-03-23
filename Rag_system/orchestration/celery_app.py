import sys
import os

# Windows: default prefork/billiard worker pool often raises PermissionError / invalid
# handle on semlocks (WinError 5 / 6). Use a single-process pool for dev.
_IS_WINDOWS = sys.platform == "win32"

# Ensure the project root (Rag_system/) is on sys.path so that sibling packages
# like `ingestion`, `core`, `agents` etc. are importable by Celery workers.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from celery import Celery

from config.settings import settings

celery_app = Celery(
    "aegis",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["orchestration.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
    # Prefork is unreliable on Windows; solo = one task at a time, same process.
    **(
        {"worker_pool": "solo", "worker_concurrency": 1}
        if _IS_WINDOWS
        else {}
    ),
)