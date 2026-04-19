# Backend API — Aman's module

FastAPI service fronting OCR, user history, language metadata, and auth. The in-browser path (onnxruntime-web) is the default for end users; this API exists for server-mediated workflows, audit trails, and any client that can't run the model locally.

## Layout

- `main.py` — FastAPI app assembly: CORS, Prometheus metrics (`ocr_requests_total`, `ocr_duration_seconds`, `ocr_queue_depth`, `model_inference_duration_seconds`), rate limiting, router wiring.
- `routers/` — `auth`, `ocr`, `history`, `languages` endpoints.
- `config.py` — settings loader (CORS origins, env-driven config).
- `db/` — database session + base models for history persistence.
- `inference/` — server-side model invocation path.
- `tasks/` — Celery task definitions for async OCR.
- `security/` — auth, file validation, hashing, rate-limiter, middleware (see `security/README.md`).

## Running

The full dev stack (API + Postgres + Redis + Celery + model worker) comes up via `start_dev.sh`, which runs `docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d`. `/health` returns `{"status": "ok"}`; `/metrics` exposes Prometheus text format.
