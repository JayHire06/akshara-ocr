import os
import re
import uuid
import aiofiles
from fastapi import UploadFile, HTTPException, status

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

MAGIC_BYTES = {
    "JPEG": [b"\xFF\xD8\xFF"],
    "PNG": [b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A"],
    "TIFF": [b"\x49\x49\x2A\x00", b"\x4D\x4D\x00\x2A"],
    "BMP": [b"\x42\x4D"]
}

def sanitize_filename(filename: str) -> str:
    """Strips path separators and special characters from filename."""
    if not filename:
        return ""
    # Only allow [a-zA-Z0-9._-]
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
    return sanitized

async def validate_and_save_upload(file: UploadFile, upload_dir: str) -> str:
    """
    Validates file size and magic bytes, generates a secure UUID filename, 
    and saves the file asynchronously.
    Returns the absolute path to the saved file.
    """
    file_size = 0
    header_chunk = await file.read(8)
    
    # Check Magic Bytes
    is_valid_type = False
    for ftype, signatures in MAGIC_BYTES.items():
        for sig in signatures:
            if header_chunk.startswith(sig):
                is_valid_type = True
                break
        if is_valid_type:
            break
            
    if not is_valid_type:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Allowed: JPEG, PNG, TIFF, BMP."
        )
        
    # Generate UUID filename
    original_ext = ""
    if file.filename:
        _, ext = os.path.splitext(file.filename)
        original_ext = sanitize_filename(ext)
        
    uuid_filename = f"{uuid.uuid4()}{original_ext}"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir, exist_ok=True)
        
    save_path = os.path.join(upload_dir, uuid_filename)
    
    # Save while tracking size
    try:
        async with aiofiles.open(save_path, 'wb') as out_file:
            await out_file.write(header_chunk)
            file_size += len(header_chunk)
            
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE_BYTES:
                    # Exceeded limit
                    await out_file.close() # Close handle before deleting
                    os.remove(save_path)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="File exceeds maximum size of 10MB."
                    )
                await out_file.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail="Error saving uploaded file")
        
    return save_path
