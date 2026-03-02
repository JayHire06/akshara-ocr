import os
import uuid
import aiofiles
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from api.db.database import get_db
from api.db.models import OCRJob, JobStatus, User
from api.db.schemas import UploadResponse
from api.tasks.ocr_task import run_ocr_job
from api.routers.auth import get_current_user

router = APIRouter(tags=["OCR"])

UPLOAD_DIR = "/tmp/uploads"

MAGIC_BYTES = {
    "png": b"\x89PNG\r\n\x1a\n",
    "jpeg": b"\xff\xd8\xff",
    "tiff_le": b"II\x2a\x00",
    "tiff_be": b"MM\x00\x2a",
    "bmp": b"BM"
}

def validate_magic_bytes(header: bytes) -> bool:
    for sig in MAGIC_BYTES.values():
        if header.startswith(sig):
            return True
    return False

@router.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    language: str = Form(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    MAX_SIZE = 10 * 1024 * 1024
    
    header = await file.read(8)
    if not validate_magic_bytes(header):
        raise HTTPException(status_code=400, detail="Invalid config or magic bytes format. Supported: PNG, JPEG, BMP, TIFF")
    
    await file.seek(0)
    
    total_size = 0
    content = bytearray()
    
    # Read in chunks to prevent large memory overhead and enforce 10MB limit
    while True:
        chunk = await file.read(1024 * 1024) # 1MB chunk
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > MAX_SIZE:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
        content.extend(chunk)
        
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    job_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename)
    if not ext:
        ext = ".png" # fallback default extension
        
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    
    async with aiofiles.open(file_path, 'wb') as out_file:
        await out_file.write(content)
        
    db_job = OCRJob(id=job_id, user_id=current_user.id, language=language, status=JobStatus.queued)
    db.add(db_job)
    await db.commit()
    
    # Enqueue Celery Task
    run_ocr_job.delay(job_id, file_path, language)
    
    return {"job_id": job_id, "status": "queued"}

@router.get("/result/{job_id}")
async def get_result(job_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(OCRJob).where(OCRJob.id == job_id)
    result = await db.execute(stmt)
    job = result.scalars().first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job.status == JobStatus.done:
        return {
            "status": job.status.value,
            "text": job.text,
            "confidence": job.confidence,
            "word_count": job.word_count,
            "processing_time_ms": job.processing_time_ms,
            "language": job.language
        }
    elif job.status in (JobStatus.processing, JobStatus.queued):
        return {"status": job.status.value}
    elif job.status == JobStatus.error:
        return {"status": job.status.value, "message": job.error_message}
