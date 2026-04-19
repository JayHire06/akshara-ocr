# DevOps — Aditya's module

Deployment, monitoring, and training-run observability.

## Files

- `prometheus.yml` — scrape config for `/metrics` on the FastAPI service. Tracks the OCR request counter/histogram pair, queue depth gauge, and model inference duration defined in `api/main.py`.
- `grafana-dashboard.json` — import into Grafana to visualise the Prometheus series.
- `training/` — training-run infrastructure.

## Local dev stack

`start_dev.sh` / `start_dev.bat` in the repo root bring up API + Postgres + Redis + Celery + model worker via `docker-compose.yml` (+ `.override.yml` for dev overrides), then start the frontend Vite dev server. Frontend at `http://localhost:5173`, API at `http://localhost:8000`.

## Training-run observability

Training scripts log live to Trackio under project `akshara-ocr`. Dashboard:

```bash
trackio show --project "akshara-ocr"
```

Historical v1–v8 runs are backfilled into the same project via `scripts/devtool/backfill_trackio.py` — run once after any v1–v8 checkpoint changes to refresh the terminal-metric panels.
