import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./test.db"
    redis_url: str = "redis://localhost:6379"
    jwt_secret: str = "test-secret-key-for-ci-only"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

# docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres
# docker run -d -p 6379:6379 redis
