from celery import Celery

celery_app = Celery(
    "aegis",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
    include=["agents.witness_agent", "orchestration.test_task"]
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
)