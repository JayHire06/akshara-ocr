import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch

from api.main import app
from api.db.database import get_db, Base

@pytest.fixture
def override_get_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    TestingSessionLocal = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async def _override_get_db():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with TestingSessionLocal() as session:
            yield session
            
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()

@pytest.fixture
def test_client(override_get_db):
    return TestClient(app)

@pytest.fixture
def auth_token(test_client):
    test_client.post("/auth/register", json={"username": "testuser", "password": "password123"})
    response = test_client.post("/auth/login", data={"username": "testuser", "password": "password123"})
    return response.json().get("access_token")

@pytest.fixture
def mock_image_blank():
    """Returns a completely white image"""
    arr = np.ones((100, 100), dtype=np.uint8) * 255
    return Image.fromarray(arr, mode='L')

@pytest.fixture
def mock_image_black():
    """Returns a completely black image"""
    arr = np.zeros((100, 100), dtype=np.uint8)
    return Image.fromarray(arr, mode='L')

@pytest.fixture
def mock_image_noise():
    """Returns an image with random noise"""
    np.random.seed(42)
    arr = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    return Image.fromarray(arr, mode='L')

@pytest.fixture
def mock_image_bimodal():
    """Returns an image with clear foreground and background for Otsu testing"""
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[20:80, 20:80] = 200 # foreground
    arr[0:20, :] = 50 # background
    arr[80:, :] = 50
    arr[:, 0:20] = 50
    arr[:, 80:] = 50
    return Image.fromarray(arr, mode='L')

@pytest.fixture
def mock_image_text_lines():
    """Returns an image simulating horizontal text lines for segmentation"""
    arr = np.ones((100, 100), dtype=np.uint8) * 255
    # Add text lines (black) ink
    arr[10:25, 10:90] = 0
    arr[40:55, 10:90] = 0
    arr[70:85, 10:90] = 0
    return Image.fromarray(arr, mode='L')
