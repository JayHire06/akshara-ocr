import os
import uuid
import asyncio
import tempfile
import shutil
import aiofiles
import threading
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.database import get_db
from api.db.models import User
from api.routers.auth import get_current_user
from api.inference.pipeline import run_ocr_pipeline

router = APIRouter(tags=["OCR"])

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "akshara_uploads")

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

# In-memory store for job results (suitable for single-process dev/testing)
job_results: dict = {}


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    language: str = Form("hin"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    MAX_SIZE = 10 * 1024 * 1024  # 10 MB

    # Validate magic bytes
    header = await file.read(8)
    if not validate_magic_bytes(header):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Supported: PNG, JPEG, BMP, TIFF"
        )
    await file.seek(0)

    # Read full file with size check
    total_size = 0
    content = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > MAX_SIZE:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
        content.extend(chunk)

    # Save to temp dir
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    job_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename or "upload.png")
    if not ext:
        ext = ".png"
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")

    async with aiofiles.open(file_path, "wb") as out_file:
        await out_file.write(bytes(content))

    # Initialise in-memory job entry
    job_results[job_id] = {
        "status": "processing",
        "text": "",
        "confidence": 0.0,
        "word_count": 0,
        "processing_time_ms": 0,
        "language": language
    }

    # Run OCR pipeline in a separate thread so it doesn't block the async event loop
    import time

    def run_with_timeout():
        print(f"OCR started for job {job_id}")
        try:
            result = run_ocr_pipeline(file_path)
            job_results[job_id].update({
                "status": result.get("status", "done"),
                "text": result.get("text", ""),
                "confidence": result.get("confidence", 0.0),
                "word_count": result.get("word_count", 0)
            })
            print(f"OCR completed for job {job_id}: {result.get('text', '')[:50]}")
        except Exception as e:
            job_results[job_id].update({
                "status": "error", 
                "text": str(e),
                "confidence": 0.0,
                "word_count": 0
            })
            print(f"OCR failed for job {job_id}: {e}")
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass

    thread = threading.Thread(target=run_with_timeout)
    thread.daemon = True
    thread.start()

    # Wait max 60 seconds then mark as timeout
    def timeout_checker():
        time.sleep(60)
        if job_results.get(job_id, {}).get("status") == "processing":
            job_results[job_id].update({
                "status": "error",
                "text": "OCR timed out after 60 seconds",
                "confidence": 0.0,
                "word_count": 0
            })
            print(f"OCR timeout triggered for job {job_id}")

    timeout_thread = threading.Thread(target=timeout_checker)
    timeout_thread.daemon = True
    timeout_thread.start()

    return {"job_id": job_id, "status": "queued"}


@router.get("/result/{job_id}")
async def get_result(job_id: str):
    if job_id not in job_results:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_results[job_id]
