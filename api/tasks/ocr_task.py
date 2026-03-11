import os
import time
import asyncio
from PIL import Image
from api.tasks.celery_app import celery_app
from api.db.database import AsyncSessionLocal
from api.db.models import OCRJob, JobStatus
from sqlalchemy import update
from api.inference.pipeline import run_ocr_pipeline

async def _update_job_status(job_id: str, status: JobStatus, **kwargs):
    async with AsyncSessionLocal() as session:
        stmt = update(OCRJob).where(OCRJob.id == job_id).values(status=status, **kwargs)
        await session.execute(stmt)
        await session.commit()

@celery_app.task(name="run_ocr_job", bind=True)
def run_ocr_job(self, job_id: str, file_path: str, language: str):
    start_time = time.time()
    
    # 1. Update DB job status to 'processing'
    asyncio.run(_update_job_status(job_id, JobStatus.processing))
    
    try:
        # 3-7. Execute pipeline (preprocess -> infer -> correct -> confidence calculation)
        result_dict = run_ocr_pipeline(file_path)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        if result_dict.get("status") == "error":
            # Handled internally but bubbled out via dict
            asyncio.run(
                _update_job_status(
                    job_id, 
                    JobStatus.error, 
                    error_message=result_dict.get("text", "Unknown error in pipeline")
                )
            )
        else:
            # 8. Update DB job with status='done', text, confidence, word_count, processing_time_ms
            asyncio.run(
                _update_job_status(
                    job_id, 
                    JobStatus.done, 
                    text=result_dict.get("text", ""), 
                    confidence=result_dict.get("confidence", 0.0), 
                    word_count=result_dict.get("word_count", 0), 
                    processing_time_ms=processing_time_ms
                )
            )
        
    except Exception as e:
        # Update DB job with status='error' and exception message
        asyncio.run(
            _update_job_status(
                job_id, 
                JobStatus.error, 
                error_message=str(e)
            )
        )
    finally:
        # 9. Delete the uploaded image file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
