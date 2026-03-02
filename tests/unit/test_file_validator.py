import pytest
from api.routers.ocr import validate_magic_bytes

def test_validate_magic_bytes_valid_jpeg():
    """Test valid JPEG magic bytes are accepted."""
    header = b"\xff\xd8\xff\xe0\x00\x10"
    assert validate_magic_bytes(header) is True

def test_validate_magic_bytes_valid_png():
    """Test valid PNG magic bytes are accepted."""
    header = b"\x89PNG\r\n\x1a\n\x00\x00"
    assert validate_magic_bytes(header) is True

def test_validate_magic_bytes_invalid():
    """Test invalid magic bytes (e.g., text file content) are rejected."""
    header = b"Hello world"
    assert validate_magic_bytes(header) is False

def test_validate_magic_bytes_short():
    """Test short header doesn't crash and is rejected."""
    header = b"\xff\xd8" # Too short to match \xff\xd8\xff
    assert validate_magic_bytes(header) is False
