from celery import Celery
from api.config import settings

celery_app = Celery(
    "ocr_worker",
    broker=settings.redis_url,
    backend=settings.redis_url
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True
)

# celery -A api.tasks.celery_app worker --loglevel=info
