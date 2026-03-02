from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime
from api.db.models import JobStatus

# --- Auth Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserCreate(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

# --- OCR Error/Response Schemas ---
class UploadResponse(BaseModel):
    job_id: str
    status: str

class JobResultResponse(BaseModel):
    status: str
    text: Optional[str] = None
    confidence: Optional[float] = None
    word_count: Optional[int] = None
    processing_time_ms: Optional[int] = None
    language: Optional[str] = None
    message: Optional[str] = None
