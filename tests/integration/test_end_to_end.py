import pytest
from unittest.mock import patch
from api.db.models import JobStatus

def test_full_pipeline_upload_to_result(test_client, auth_token):
    """
    Full pipeline test: upload image -> get text result.
    Simulates the end-to-end user flow: uploading an image and polling for the result.
    """
    # 1. Upload image
    with patch('api.routers.ocr.run_ocr_job.delay') as mock_delay:
        file_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"A" * 100
        files = {'file': ('document.png', file_content, 'image/png')}
        data = {'language': 'hi'}
        headers = {'Authorization': f'Bearer {auth_token}'}
        
        upload_res = test_client.post("/upload", files=files, data=data, headers=headers)
        assert upload_res.status_code == 200, upload_res.text
        
        # Verify job was created
        job_data = upload_res.json()
        assert "job_id" in job_data
        job_id = job_data["job_id"]
        assert job_data["status"] == "queued"
        
        # Verify background task was called
        assert mock_delay.called
        
    # 2. Get Result (Polling)
    # The database was updated by the /upload endpoint, so it should exist locally
    poll_res = test_client.get(f"/result/{job_id}", headers=headers)
    assert poll_res.status_code == 200
    
    poll_data = poll_res.json()
    assert "status" in poll_data
    
    # It should still be queued since the celery worker is mocked and didn't update the DB
    assert poll_data["status"] == "queued"
