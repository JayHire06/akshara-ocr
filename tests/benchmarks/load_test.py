from locust import HttpUser, task, between
import os
import time

class OCRUser(HttpUser):
    wait_time = between(1, 5)
    
    def on_start(self):
        """Login or register on start to get an auth token for API calls."""
        self.username = f"load_user_{os.urandom(4).hex()}"
        self.password = "load_password"
        
        # Register user
        self.client.post("/auth/register", json={"username": self.username, "password": self.password})
        
        # Login
        res = self.client.post(
            "/auth/login", 
            data={"username": self.username, "password": self.password}
        )
        if res.status_code == 200:
            self.token = res.json().get("access_token")
        else:
            self.token = ""

    @task
    def upload_and_poll(self):
        """
        Simulate a user uploading an image, pasting the task to the backend,
        and polling until the result is returned. Targets <5s 95th percentile and <1% error rate.
        """
        headers = {"Authorization": f"Bearer {self.token}"}
        
        # 1. Upload mock PNG image
        file_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"A" * 1024 # 1KB image
        files = {'file': ('test.png', file_content, 'image/png')}
        data = {'language': 'hi'}
        
        with self.client.post("/upload", files=files, data=data, headers=headers, catch_response=True) as upload_response:
            if upload_response.status_code != 200:
                upload_response.failure(f"Upload failed with HTTP {upload_response.status_code}")
                return
            job_id = upload_response.json().get("job_id")
            
        if not job_id:
            return
            
        # 2. Poll for the result
        for _ in range(15): # Poll up to 15 times (15 seconds total)
            time.sleep(1) # wait 1s between polls
            with self.client.get(f"/result/{job_id}", headers=headers, catch_response=True) as poll_response:
                if poll_response.status_code == 200:
                    status = poll_response.json().get("status")
                    if status == "done":
                        poll_response.success()
                        return
                    elif status == "error":
                        poll_response.failure("OCR job resulted in an error status")
                        return
                    # if queued or processing, we loop again
                else:
                    poll_response.failure(f"Polling failed with HTTP {poll_response.status_code}")
                    return
        
        # If we reach here, polling timed out
        self.environment.events.request.fire(
            request_type="GET",
            name="/result/[job_id] (timeout)",
            response_time=15000,
            response_length=0,
            exception=TimeoutError("Polling timed out after 15s")
        )

# Command to run (output HTML report): 
# locust -f tests/benchmarks/load_test.py --headless -u 100 -r 10 --run-time 5m --html tests/benchmarks/load_report.html
