from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest

from api.routers import auth, ocr, history, languages
from api.db.database import engine, Base
from api.config import settings

try:
    from slowapi.errors import RateLimitExceeded
    from slowapi import _rate_limit_exceeded_handler
    from api.security.rate_limiter import limiter
except ImportError:
    RateLimitExceeded = None
    _rate_limit_exceeded_handler = None
    limiter = None

app = FastAPI(title="Akshara OCR API")

allow_origins = [
    origin.strip()
    for origin in settings.cors_allow_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=settings.cors_allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if limiter is not None:
    app.state.limiter = limiter

if RateLimitExceeded is not None and _rate_limit_exceeded_handler is not None:
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.on_event("startup")
async def startup():
    Base.metadata.create_all(bind=engine)
# Prometheus Metrics
ocr_requests_total = Counter(
    "ocr_requests_total",
    "Total number of OCR requests",
    ["language", "status"]
)

ocr_duration_seconds = Histogram(
    "ocr_duration_seconds",
    "Time taken to process an OCR request",
    ["language"]
)

ocr_queue_depth = Gauge(
    "ocr_queue_depth",
    "Current number of OCR tasks in the queue"
)

model_inference_duration_seconds = Histogram(
    "model_inference_duration_seconds",
    "Time taken for model inference"
)

app.include_router(auth.router)
app.include_router(ocr.router)
app.include_router(history.router)
app.include_router(languages.router)

@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type="text/plain")

@app.get("/health")
def health_check():
    return {"status": "ok"}
