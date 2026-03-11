import os
from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete
import secrets
import hashlib

from api.db.models import User, RefreshToken

# Extract JWT_SECRET
JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    from api.config import settings
    JWT_SECRET = getattr(settings, "jwt_secret", "")

if len(JWT_SECRET) < 32:
    raise ValueError("JWT_SECRET environment variable must be at minimum 32 characters long.")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7


def create_access_token(user_id: int, email: str) -> str:
    """Generates a short-lived access token."""
    expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire_time = datetime.now(timezone.utc) + expires_delta
    to_encode = {
        "sub": str(user_id),
        "email": email,
        "exp": expire_time.timestamp(),
        "type": "access"
    }
    return jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)


def hash_refresh_token(token: str) -> str:
    """Hashes the refresh token for secure database storage."""
    return hashlib.sha256(token.encode()).hexdigest()


async def create_refresh_token(user_id: int, db: AsyncSession) -> str:
    """Generates a cryptographically strong refresh token and saves its hash."""
    raw_token = secrets.token_urlsafe(32)
    hashed_token = hash_refresh_token(raw_token)
    
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    db_token = RefreshToken(
        user_id=user_id,
        hashed_token=hashed_token,
        expires_at=expires_at,
        is_revoked=False
    )
    db.add(db_token)
    await db.commit()
    
    return raw_token


async def rotate_refresh_token(old_refresh_token: str, db: AsyncSession):
    """
    Validates the old refresh token.
    If valid: revokes old, creates new access + refresh pair.
    If reused: theft detection -> revoke all user's tokens!
    """
    hashed_old_token = hash_refresh_token(old_refresh_token)
    stmt = select(RefreshToken).where(RefreshToken.hashed_token == hashed_old_token)
    result = await db.execute(stmt)
    db_token = result.scalars().first()
    
    # Check if token exists
    if not db_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # Check if token expired
    if db_token.expires_at.tzinfo is None:
        db_token.expires_at = db_token.expires_at.replace(tzinfo=timezone.utc)
    if db_token.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # User theft detection 
    if db_token.is_revoked:
        # Threat detected! Revoke ALL tokens for this user.
        await db.execute(
            delete(RefreshToken).where(RefreshToken.user_id == db_token.user_id)
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token reuse detected. All sessions revoked. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # All checks passed, perform rotation
    db_token.is_revoked = True
    
    # Get user to create new tokens
    user_stmt = select(User).where(User.id == db_token.user_id)
    user_result = await db.execute(user_stmt)
    user = user_result.scalars().first()
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
        
    new_access_token = create_access_token(user_id=user.id, email=user.username)
    new_refresh_token = await create_refresh_token(user_id=user.id, db=db)
    
    return new_access_token, new_refresh_token
