from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc

from api.db.database import get_db
from api.db.models import OCRJob, User
from api.routers.auth import get_current_user

router = APIRouter(prefix="/history", tags=["History"])

@router.get("/")
async def get_history(
    skip: int = Query(0, ge=0), 
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    stmt = select(OCRJob).where(OCRJob.user_id == current_user.id).order_by(desc(OCRJob.created_at)).offset(skip).limit(limit)
    result = await db.execute(stmt)
    jobs = result.scalars().all()
    
    return [
        {
            "job_id": job.id,
            "status": job.status.value,
            "language": job.language,
            "created_at": job.created_at,
            "updated_at": job.updated_at
        } for job in jobs
    ]
