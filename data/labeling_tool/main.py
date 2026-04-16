# data/labeling_tool/main.py
from __future__ import annotations

import datetime as _dt
import os
import sqlite3
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

_MODULE_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _MODULE_DIR / "templates"
_LEGACY_IMAGES_DIR = _MODULE_DIR / "images_to_label"
_LEGACY_LABELS_FILE = _MODULE_DIR / "manual_labels.csv"

_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
_LEGACY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
app.mount("/images", StaticFiles(directory=str(_LEGACY_IMAGES_DIR)), name="images")


def _legacy_unlabeled() -> list[str]:
    import csv
    labeled: set[str] = set()
    if _LEGACY_LABELS_FILE.exists():
        with open(_LEGACY_LABELS_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    labeled.add(row[0])
    all_images = [
        f for f in os.listdir(_LEGACY_IMAGES_DIR) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    return [img for img in all_images if img not in labeled]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    unlabeled = _legacy_unlabeled()
    current = unlabeled[0] if unlabeled else None
    return templates.TemplateResponse(
        request,
        "index.html",
        {"image": current, "remaining": len(unlabeled)},
    )


@app.post("/submit", response_class=HTMLResponse)
async def submit(
    request: Request,
    filename: str = Form(...),
    text: str = Form(...),
    language: str = Form(...),
):
    import csv
    if not _LEGACY_LABELS_FILE.exists():
        with open(_LEGACY_LABELS_FILE, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "text", "language", "font", "is_synthetic"])
    with open(_LEGACY_LABELS_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([filename, text, language, "manual", False])
    return await index(request)


AUTO_DB_ENV_VAR = "AUTO_LABELED_WORK_DB"
DEFAULT_AUTO_DB = Path("data/external/auto_labeled/work.db")


def _auto_db_path() -> Path:
    env = os.environ.get(AUTO_DB_ENV_VAR)
    return Path(env) if env else DEFAULT_AUTO_DB


def _auto_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_auto_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def _next_candidate(conn: sqlite3.Connection) -> sqlite3.Row | None:
    row = conn.execute(
        """
        SELECT w.id, w.text_nfc, w.conf, p.source
        FROM words w JOIN pages p ON p.id = w.page_id
        WHERE w.status = 'exported_qa_candidate' AND w.qa_decision IS NULL
        ORDER BY (
            SELECT COUNT(*)
            FROM words w2 JOIN pages p2 ON p2.id = w2.page_id
            WHERE w2.status='exported_qa_candidate'
              AND w2.qa_decision IS NOT NULL
              AND p2.source = p.source
        ) ASC, RANDOM()
        LIMIT 1
        """
    ).fetchone()
    return row


@app.get("/qa")
async def qa_next(request: Request):
    conn = _auto_conn()
    try:
        row = _next_candidate(conn)
    finally:
        conn.close()
    if row is None:
        return templates.TemplateResponse(
            request,
            "qa.html",
            {"word": None, "done": True},
        )
    resp = templates.TemplateResponse(
        request,
        "qa.html",
        {"word": dict(row), "done": False},
    )
    resp.headers["x-qa-word-id"] = str(row["id"])
    return resp


@app.post("/qa/decide")
async def qa_decide(
    word_id: int = Form(...),
    action: str = Form(...),
    edited_text: str = Form(""),
):
    if action not in ("accept", "edit", "reject"):
        return JSONResponse({"error": "bad action"}, status_code=400)
    conn = _auto_conn()
    try:
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        if action == "edit":
            conn.execute(
                "UPDATE words SET qa_decision=?, qa_edited_text=?, qa_decided_at=? WHERE id=?",
                ("edit", edited_text, now, word_id),
            )
        else:
            conn.execute(
                "UPDATE words SET qa_decision=?, qa_decided_at=? WHERE id=?",
                (action, now, word_id),
            )
        conn.commit()
    finally:
        conn.close()
    return Response(status_code=303, headers={"location": "/qa"})


@app.get("/qa/stats")
async def qa_stats():
    conn = _auto_conn()
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM words WHERE status='exported_qa_candidate'"
        ).fetchone()[0]
        decided = conn.execute(
            "SELECT COUNT(*) FROM words"
            " WHERE status='exported_qa_candidate' AND qa_decision IS NOT NULL"
        ).fetchone()[0]
        per_source = {
            row[0]: {"total": row[1], "decided": row[2]}
            for row in conn.execute(
                """
                SELECT p.source,
                       COUNT(*) AS total,
                       SUM(CASE WHEN w.qa_decision IS NOT NULL THEN 1 ELSE 0 END) AS decided
                FROM words w JOIN pages p ON p.id = w.page_id
                WHERE w.status='exported_qa_candidate'
                GROUP BY p.source
                """
            )
        }
    finally:
        conn.close()
    return {"total": total, "decided": decided, "per_source": per_source}
