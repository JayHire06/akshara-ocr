# Akshara OCR

Akshara OCR is an intelligent pipeline and application designed to recognize text accurately from complex documents. 

## Development Prerequisites

To successfully run Akshara OCR on your local machine, ensure you have the following installed:

1. **Docker / Docker Desktop**: Required for spinning up the core backend services (PostgreSQL, Redis, Celery Workers, and the Model Server).
2. **Node.js & npm**: Required to install dependencies and run the frontend interface.

## Quickstart

Start the entire stack (both frontend and backend) simultaneously with a single command:

**On Windows:**
```cmd
.\start_dev.bat
```

**On Linux / macOS / Git Bash:**
```bash
bash ./start_dev.sh
```

By default, the backend API operates on `http://localhost:8000` while the frontend is served at `http://localhost:5173`.