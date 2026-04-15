# External Auto-Labeled Data Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a scrape → cloud-OCR-label → filter → QA pipeline that produces a v8-compatible training stage (~150–200K rows) and a 2,000-row human-verified test suite from Tier-1 + Tier-4 public-domain Hindi printed content, fully enforcing the §5 text-disjoint policy from the spec.

**Architecture:** Four staged Python scripts orchestrated by a shell wrapper, with a SQLite work queue (`data/external/auto_labeled/work.db`) as the single source of truth. Filtering and text-disjoint enforcement happen as SQL queries against the queue. The QA verification tool is an extension of the existing `data/labeling_tool/` FastAPI app with new endpoints and a keyboard-driven template. Tests live under the repo's existing `tests/unit/` and `tests/integration/` directories to match repo convention.

**Tech Stack:** Python 3.14 (repo venv), SQLite 3.25+ (window functions required by §5.3 SQL), Playwright for Wikipedia/Wikisource rendering, `pdf2image` for gov PDFs, `httpx` for archive.org API, `tenacity` for retries, `asyncio` for labeler concurrency, `python-Levenshtein` for audit, FastAPI + Jinja2 (existing `data/labeling_tool/`), `pytest` for tests.

**Spec:** [`docs/superpowers/specs/2026-04-16-external-auto-labeled-data-design.md`](../specs/2026-04-16-external-auto-labeled-data-design.md)

---

## File Structure

### Files to create

| Path | Purpose |
|---|---|
| `data/external/auto_labeled/README.md` | Pipeline docs and operator runbook |
| `scripts/data/auto_labeled_schema.sql` | `work.db` table definitions |
| `scripts/data/auto_labeled_common.py` | Shared helpers: DB connection, NFC wrappers, atomic write, string-set loaders |
| `scripts/data/auto_labeled_provider.py` | `Provider` ABC + `AzureReadProvider` + `FakeProvider` + factory |
| `scripts/data/scrape_external_pages.py` | Scraper CLI for all four source tiers |
| `scripts/data/label_with_cloud_ocr.py` | Labeler CLI: async drain pages, label via provider, write words rows |
| `scripts/data/filter_and_crop_words.py` | Filter CLI: confidence, OOV, bbox, dup, text-disjoint (policy B), crop, emit labels |
| `scripts/data/build_auto_labeled_stage.py` | Thin wrapper that produces a v8-compatible stage directory |
| `scripts/data/run_auto_labeled_pipeline.sh` | Orchestrator shell script for end-to-end runs |
| `data/labeling_tool/templates/qa.html` | QA verification view template, keyboard-driven |
| `tests/unit/test_auto_labeled_common.py` | Tests for NFC, atomic write, string-set loader |
| `tests/unit/test_auto_labeled_provider.py` | Tests for fake provider and retry wrapper |
| `tests/unit/test_auto_labeled_filter.py` | Tests for every filter stage (confidence, OOV, bbox, dup, policy B, QA selection, audits) |
| `tests/unit/test_auto_labeled_qa_export.py` | Tests for verified-test-labels export (decision filtering, SHA256) |
| `tests/integration/test_auto_labeled_pipeline_smoke.py` | End-to-end smoke against fake provider and 3 fixture pages |
| `tests/fixtures/auto_labeled/*.png` | Small synthetic page fixtures |
| `tests/fixtures/auto_labeled/fake_responses.json` | Canned provider responses used by fake provider |

### Files to modify

| Path | Change |
|---|---|
| `data/labeling_tool/main.py` | Add `/qa`, `/qa/decide`, `/qa/stats` endpoints (preserve existing `/` and `/submit`) |
| `model/benchmark_eval.py` | Register `auto_labeled_verified_printed` suite (purely additive) |
| `scripts/training/train_v9.py` | Insert `stage_auto_labeled_printed` between Stage 1 synthetic and Stage 2 Mozhi in the curriculum |

### Test routing note

The spec §8.1 suggested `scripts/data/tests/` as the test location, but the repo convention (verified against `tests/unit/test_v8_dataset_prep.py`) uses a project-level `tests/` directory. This plan routes tests to `tests/unit/` and `tests/integration/` to match existing practice. This is the only deviation from the spec's literal wording, and it trades spec literalness for repo consistency.

### Prerequisites (one-time, not a task)

Before Task 1, verify the repo venv has: `playwright`, `pdf2image`, `httpx`, `tenacity`, `python-Levenshtein`, `pytest`, `pillow`, `azure-ai-documentintelligence` (or the provider SDK matching your chosen provider). Install missing packages with `pip install -r requirements.txt` and, for Playwright, `playwright install chromium`. If you don't have Azure credentials yet, Tasks 1–16 still run end-to-end against the fake provider — only Task 17's manual integration check requires real credentials.

---

## Task 1: Scaffold directory tree, schema, and common helpers

**Files:**
- Create: `data/external/auto_labeled/README.md`
- Create: `scripts/data/auto_labeled_schema.sql`
- Create: `scripts/data/auto_labeled_common.py`
- Create: `tests/unit/test_auto_labeled_common.py`

This task establishes the foundation. Everything downstream imports from `auto_labeled_common.py` and uses the schema from `auto_labeled_schema.sql`. The test verifies the three pure-logic helpers before any pipeline code exists.

- [ ] **Step 1: Create the data directory and README**

Run:

```bash
mkdir -p data/external/auto_labeled/raw/{wikipedia,wikisource,gov_pdf,archive_org}
mkdir -p data/external/auto_labeled/normalized/crops
mkdir -p data/external/auto_labeled/manifests
```

Then create `data/external/auto_labeled/README.md`:

```markdown
# Auto-Labeled External Data Pipeline

See: `docs/superpowers/specs/2026-04-16-external-auto-labeled-data-design.md`

This directory is populated by `scripts/data/run_auto_labeled_pipeline.sh`.

## Layout

- `raw/<source>/` — scraped page PNGs
- `work.db` — SQLite work queue (source of truth during a run)
- `normalized/crops/` — cropped word PNGs produced by the filter stage
- `normalized/*.txt` — pipe-format label files
- `manifests/manifest.json` — provenance, audit counts, file hashes

## Quick start

bash scripts/data/run_auto_labeled_pipeline.sh --source wikipedia --limit 5

The full run is documented in the spec's §9 integration section.
```

- [ ] **Step 2: Create the SQL schema file**

Create `scripts/data/auto_labeled_schema.sql`:

```sql
-- Auto-labeled data pipeline work queue schema.
-- See docs/superpowers/specs/2026-04-16-external-auto-labeled-data-design.md §3.1

CREATE TABLE IF NOT EXISTS pages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL CHECK (source IN ('wikipedia','wikisource','gov_pdf','archive_org')),
    uri          TEXT NOT NULL,
    local_path   TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('pending','labeled','fetch_failed','label_failed')),
    error        TEXT
);

CREATE INDEX IF NOT EXISTS idx_pages_status ON pages(status);
CREATE INDEX IF NOT EXISTS idx_pages_source ON pages(source);

CREATE TABLE IF NOT EXISTS words (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id         INTEGER NOT NULL REFERENCES pages(id),
    x               INTEGER NOT NULL,
    y               INTEGER NOT NULL,
    w               INTEGER NOT NULL,
    h               INTEGER NOT NULL,
    text_nfc        TEXT NOT NULL,
    conf            REAL NOT NULL,
    provider        TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN (
                        'labeled','kept','exported_qa_candidate',
                        'filtered_low_conf','filtered_oov','filtered_invalid_bbox',
                        'filtered_dup','filtered_text_leak'
                    )),
    filter_reason   TEXT,
    qa_decision     TEXT CHECK (qa_decision IN ('accept','edit','reject')),
    qa_edited_text  TEXT,
    qa_decided_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_words_status ON words(status);
CREATE INDEX IF NOT EXISTS idx_words_text_nfc ON words(text_nfc);
CREATE INDEX IF NOT EXISTS idx_words_page_id ON words(page_id);

CREATE TABLE IF NOT EXISTS manifest (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

- [ ] **Step 3: Write the failing test for common helpers**

Create `tests/unit/test_auto_labeled_common.py`:

```python
from __future__ import annotations

import sqlite3
import tempfile
import unicodedata
from pathlib import Path

import pytest

from scripts.data.auto_labeled_common import (
    atomic_write_text,
    load_text_strings,
    normalize_text,
    open_work_db,
)


def test_normalize_text_applies_nfc_and_strip():
    # Decomposed form of "भा" (BHA): क्ष written as क + ् + ष
    decomposed = "क\u094dष"
    composed = unicodedata.normalize("NFC", decomposed)
    assert normalize_text(f"  {decomposed}  ") == composed


def test_normalize_text_preserves_zwj():
    zwj_string = "क\u200dष"
    assert normalize_text(zwj_string) == unicodedata.normalize("NFC", zwj_string)


def test_atomic_write_text_is_atomic(tmp_path: Path):
    target = tmp_path / "out.txt"
    atomic_write_text(target, "hello\nworld\n")
    assert target.read_text(encoding="utf-8") == "hello\nworld\n"
    # No stray temp files left behind.
    leftovers = [p for p in tmp_path.iterdir() if p.name != target.name]
    assert leftovers == []


def test_load_text_strings_reads_pipe_format(tmp_path: Path):
    f = tmp_path / "labels.txt"
    f.write_text("/a/b.png|भारत\n/c/d.png|हिंदी\n", encoding="utf-8")
    strings = load_text_strings([f])
    assert strings == {"भारत", "हिंदी"}


def test_load_text_strings_normalizes(tmp_path: Path):
    f = tmp_path / "labels.txt"
    decomposed = "क\u094dष"
    f.write_text(f"/x.png|{decomposed}\n", encoding="utf-8")
    strings = load_text_strings([f])
    assert strings == {unicodedata.normalize("NFC", decomposed)}


def test_load_text_strings_skips_blank_and_malformed(tmp_path: Path):
    f = tmp_path / "labels.txt"
    f.write_text("\n/a.png|x\nmalformed_no_pipe\n\n", encoding="utf-8")
    strings = load_text_strings([f])
    assert strings == {"x"}


def test_open_work_db_initializes_schema(tmp_path: Path):
    db_path = tmp_path / "work.db"
    conn = open_work_db(db_path)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"pages", "words", "manifest"} <= tables
    conn.close()
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `pytest tests/unit/test_auto_labeled_common.py -v`
Expected: All tests FAIL with `ModuleNotFoundError: No module named 'scripts.data.auto_labeled_common'`.

- [ ] **Step 5: Implement common helpers**

Create `scripts/data/auto_labeled_common.py`:

```python
from __future__ import annotations

import os
import sqlite3
import tempfile
import unicodedata
from pathlib import Path
from typing import Iterable

SCHEMA_PATH = Path(__file__).parent / "auto_labeled_schema.sql"


def normalize_text(text: str) -> str:
    """NFC-normalize and strip. Canonical for this pipeline."""
    return unicodedata.normalize("NFC", text).strip()


def atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically via temp file + os.replace.

    Guarantees readers never see a half-written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def load_text_strings(paths: Iterable[Path]) -> set[str]:
    """Read one or more pipe-format label files and return the set of NFC-normalized text strings.

    Format: each line is `image_path|text`. Blank lines and malformed rows are skipped.
    """
    result: set[str] = set()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or "|" not in line:
                    continue
                _, _, text = line.partition("|")
                text = normalize_text(text)
                if text:
                    result.add(text)
    return result


def open_work_db(path: Path) -> sqlite3.Connection:
    """Open (or create) the work queue DB and ensure the schema is present."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    return conn
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/unit/test_auto_labeled_common.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add data/external/auto_labeled/README.md scripts/data/auto_labeled_schema.sql scripts/data/auto_labeled_common.py tests/unit/test_auto_labeled_common.py
git commit -m "feat(auto-labeled): scaffold work queue schema and common helpers"
```

---

## Task 2: Provider interface and fake provider

**Files:**
- Create: `scripts/data/auto_labeled_provider.py`
- Create: `tests/fixtures/auto_labeled/fake_responses.json`
- Create: `tests/unit/test_auto_labeled_provider.py`

The provider interface is what makes the labeler testable without network. The fake provider returns canned responses loaded from a JSON file; the Azure provider (added here too, behind the same interface) talks to the real API. Every later test uses the fake provider.

- [ ] **Step 1: Create the fake-response fixture**

Create `tests/fixtures/auto_labeled/fake_responses.json`:

```json
{
  "page_wiki_001.png": [
    {"x": 12, "y": 20, "w": 80, "h": 24, "text": "भारत", "conf": 0.97},
    {"x": 98, "y": 20, "w": 90, "h": 24, "text": "हिंदी", "conf": 0.93},
    {"x": 12, "y": 50, "w": 110, "h": 24, "text": "विकिपीडिया", "conf": 0.88}
  ],
  "page_wiki_002.png": [
    {"x": 20, "y": 30, "w": 60, "h": 22, "text": "कहानी", "conf": 0.95},
    {"x": 90, "y": 30, "w": 70, "h": 22, "text": "लिखो", "conf": 0.91}
  ],
  "page_empty.png": []
}
```

- [ ] **Step 2: Write the failing test**

Create `tests/unit/test_auto_labeled_provider.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.data.auto_labeled_provider import (
    FakeProvider,
    WordBox,
    get_provider,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "auto_labeled"
FAKE_JSON = FIXTURES / "fake_responses.json"


def test_fake_provider_returns_canned_boxes():
    provider = FakeProvider(responses_path=FAKE_JSON)
    boxes = provider.label_page(Path("page_wiki_001.png"))
    assert len(boxes) == 3
    assert boxes[0] == WordBox(x=12, y=20, w=80, h=24, text="भारत", conf=0.97)


def test_fake_provider_returns_empty_for_empty_page():
    provider = FakeProvider(responses_path=FAKE_JSON)
    assert provider.label_page(Path("page_empty.png")) == []


def test_fake_provider_raises_for_unknown_page():
    provider = FakeProvider(responses_path=FAKE_JSON)
    with pytest.raises(KeyError):
        provider.label_page(Path("nonexistent.png"))


def test_fake_provider_nfc_normalizes_text(tmp_path):
    f = tmp_path / "fake.json"
    f.write_text(json.dumps({
        "p.png": [{"x": 0, "y": 0, "w": 1, "h": 1, "text": "क\u094dष", "conf": 0.9}]
    }))
    provider = FakeProvider(responses_path=f)
    boxes = provider.label_page(Path("p.png"))
    assert boxes[0].text == "क्ष"  # NFC form


def test_get_provider_returns_fake_by_name(tmp_path):
    provider = get_provider("fake", fake_responses_path=FAKE_JSON)
    assert isinstance(provider, FakeProvider)


def test_get_provider_raises_for_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("nope")
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/unit/test_auto_labeled_provider.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement the provider module**

Create `scripts/data/auto_labeled_provider.py`:

```python
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from scripts.data.auto_labeled_common import normalize_text


@dataclass(frozen=True)
class WordBox:
    x: int
    y: int
    w: int
    h: int
    text: str
    conf: float


class Provider(ABC):
    name: str = "abstract"

    @abstractmethod
    def label_page(self, image_path: Path) -> list[WordBox]:
        """Label a single page image. Return a list of word boxes.

        Implementations must NFC-normalize the returned text and must not
        return boxes with empty or whitespace-only text.
        """


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, responses_path: Path):
        self._responses: dict[str, list[dict]] = json.loads(
            Path(responses_path).read_text(encoding="utf-8")
        )

    def label_page(self, image_path: Path) -> list[WordBox]:
        key = Path(image_path).name
        if key not in self._responses:
            raise KeyError(f"fake provider has no response for: {key}")
        raw = self._responses[key]
        out: list[WordBox] = []
        for b in raw:
            text = normalize_text(b["text"])
            if not text:
                continue
            out.append(
                WordBox(
                    x=int(b["x"]),
                    y=int(b["y"]),
                    w=int(b["w"]),
                    h=int(b["h"]),
                    text=text,
                    conf=float(b["conf"]),
                )
            )
        return out


class AzureReadProvider(Provider):
    name = "azure"

    def __init__(self, endpoint: str, key: str):
        # Deferred import so tests run without the SDK installed.
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential

        self._client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

    def label_page(self, image_path: Path) -> list[WordBox]:
        with Path(image_path).open("rb") as f:
            poller = self._client.begin_analyze_document(
                model_id="prebuilt-read", body=f
            )
        result = poller.result()
        out: list[WordBox] = []
        for page in result.pages or []:
            for word in page.words or []:
                text = normalize_text(word.content or "")
                if not text:
                    continue
                poly = word.polygon  # flat list [x1,y1,x2,y2,x3,y3,x4,y4]
                xs = poly[0::2]
                ys = poly[1::2]
                x, y = int(min(xs)), int(min(ys))
                w, h = int(max(xs) - min(xs)), int(max(ys) - min(ys))
                out.append(
                    WordBox(x=x, y=y, w=w, h=h, text=text, conf=float(word.confidence))
                )
        return out


def get_provider(name: str, **kwargs) -> Provider:
    if name == "fake":
        return FakeProvider(responses_path=kwargs["fake_responses_path"])
    if name == "azure":
        import os
        endpoint = kwargs.get("azure_endpoint") or os.environ["AZURE_DOC_INTEL_ENDPOINT"]
        key = kwargs.get("azure_key") or os.environ["AZURE_DOC_INTEL_KEY"]
        return AzureReadProvider(endpoint=endpoint, key=key)
    raise ValueError(f"unknown provider: {name}")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/unit/test_auto_labeled_provider.py -v`
Expected: All 6 tests PASS. (The Azure provider is not exercised; its import is deferred to instantiation.)

- [ ] **Step 6: Commit**

```bash
git add scripts/data/auto_labeled_provider.py tests/fixtures/auto_labeled/fake_responses.json tests/unit/test_auto_labeled_provider.py
git commit -m "feat(auto-labeled): provider interface with fake and Azure impls"
```

---

## Task 3: Scraper CLI — Wikipedia source

**Files:**
- Create: `scripts/data/scrape_external_pages.py`
- Modify (in later tasks): same file for additional sources
- Create: `tests/unit/test_auto_labeled_scraper_wikipedia.py`

The scraper has one main entry point and one function per source tier. This task lays down the argparse shell, the DB insert logic, and the Wikipedia source. Later tasks add the other three sources to the same file.

The test here uses monkeypatching to stub Playwright rather than actually launching a browser, so it runs in CI without any extra setup.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_auto_labeled_scraper_wikipedia.py`:

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.data.auto_labeled_common import open_work_db
from scripts.data.scrape_external_pages import scrape_wikipedia


def test_scrape_wikipedia_inserts_pending_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "work.db"
    raw_dir = tmp_path / "raw" / "wikipedia"
    raw_dir.mkdir(parents=True)

    fake_urls = ["https://hi.wikipedia.org/wiki/एक", "https://hi.wikipedia.org/wiki/दो"]

    def fake_random_article_urls(limit):
        return fake_urls[:limit]

    def fake_screenshot(url, out_path):
        out_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._random_wikipedia_article_urls",
        fake_random_article_urls,
    )
    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._screenshot_url",
        fake_screenshot,
    )

    conn = open_work_db(db_path)
    scrape_wikipedia(conn=conn, limit=2, raw_dir=raw_dir)
    conn.commit()

    rows = list(conn.execute("SELECT source, uri, status FROM pages ORDER BY id"))
    assert len(rows) == 2
    assert all(r[0] == "wikipedia" for r in rows)
    assert rows[0][1] == fake_urls[0]
    assert all(r[2] == "pending" for r in rows)

    # Files were actually written.
    paths = [Path(r[0]) for r in conn.execute("SELECT local_path FROM pages")]
    assert all(p.exists() for p in paths)


def test_scrape_wikipedia_marks_blank_renders_as_fetch_failed(tmp_path, monkeypatch):
    db_path = tmp_path / "work.db"
    raw_dir = tmp_path / "raw" / "wikipedia"
    raw_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._random_wikipedia_article_urls",
        lambda limit: ["https://hi.wikipedia.org/wiki/Blank"],
    )
    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._screenshot_url",
        lambda url, out_path: out_path.write_bytes(b""),
    )
    # Force the non-blank check to report blank.
    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._is_non_blank_png",
        lambda path: False,
    )

    conn = open_work_db(db_path)
    scrape_wikipedia(conn=conn, limit=1, raw_dir=raw_dir)
    conn.commit()

    status = conn.execute("SELECT status FROM pages").fetchone()[0]
    assert status == "fetch_failed"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_auto_labeled_scraper_wikipedia.py -v`
Expected: FAIL with `ImportError` or `AttributeError`.

- [ ] **Step 3: Implement the scraper CLI skeleton and Wikipedia source**

Create `scripts/data/scrape_external_pages.py`:

```python
from __future__ import annotations

import argparse
import datetime as _dt
import random
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Iterable

import httpx
from PIL import Image

from scripts.data.auto_labeled_common import open_work_db

WIKIPEDIA_API = "https://hi.wikipedia.org/w/api.php"
WIKIPEDIA_BASE = "https://hi.wikipedia.org/wiki/"

SOURCES = ("wikipedia", "wikisource", "gov_pdf", "archive_org")


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _is_non_blank_png(path: Path) -> bool:
    """Return True if the PNG is openable and has at least 5% non-white pixels."""
    try:
        with Image.open(path) as im:
            im = im.convert("L")
            pixels = list(im.getdata())
    except Exception:
        return False
    if not pixels:
        return False
    non_white = sum(1 for p in pixels if p < 240)
    return (non_white / len(pixels)) >= 0.05


def _insert_page(conn: sqlite3.Connection, source: str, uri: str, local_path: Path, status: str, error: str | None = None) -> None:
    conn.execute(
        "INSERT INTO pages (source, uri, local_path, fetched_at, status, error)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (source, uri, str(local_path), _now_iso(), status, error),
    )


# --- Wikipedia ------------------------------------------------------------------


def _random_wikipedia_article_urls(limit: int) -> list[str]:
    """Use the MediaWiki API to pick `limit` random article URLs from hi.wikipedia."""
    with httpx.Client(timeout=30.0) as client:
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": min(limit, 10),
        }
        urls: list[str] = []
        while len(urls) < limit:
            resp = client.get(WIKIPEDIA_API, params=params)
            resp.raise_for_status()
            data = resp.json()
            for item in data["query"]["random"]:
                title = item["title"].replace(" ", "_")
                urls.append(WIKIPEDIA_BASE + title)
                if len(urls) >= limit:
                    break
    return urls[:limit]


def _screenshot_url(url: str, out_path: Path) -> None:
    """Render a URL with Playwright and save a full-page screenshot."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 2000})
            page.goto(url, wait_until="networkidle", timeout=60_000)
            page.screenshot(path=str(out_path), full_page=True)
        finally:
            browser.close()


def scrape_wikipedia(conn: sqlite3.Connection, limit: int, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    urls = _random_wikipedia_article_urls(limit)
    for url in urls:
        out = raw_dir / f"{uuid.uuid4().hex}.png"
        try:
            _screenshot_url(url, out)
            if not _is_non_blank_png(out):
                _insert_page(conn, "wikipedia", url, out, "fetch_failed", error="blank_render")
                continue
            _insert_page(conn, "wikipedia", url, out, "pending")
        except Exception as exc:
            _insert_page(conn, "wikipedia", url, out, "fetch_failed", error=str(exc)[:500])


# --- CLI entry point -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scrape external printed Hindi pages into the work queue.")
    p.add_argument("--source", required=True, choices=SOURCES)
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--db", type=Path, default=Path("data/external/auto_labeled/work.db"))
    p.add_argument("--raw-root", type=Path, default=Path("data/external/auto_labeled/raw"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    conn = open_work_db(args.db)
    try:
        raw_dir = args.raw_root / args.source
        if args.source == "wikipedia":
            scrape_wikipedia(conn=conn, limit=args.limit, raw_dir=raw_dir)
        else:
            print(f"source {args.source} not yet implemented", file=sys.stderr)
            return 2
        conn.commit()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_auto_labeled_scraper_wikipedia.py -v`
Expected: Both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/data/scrape_external_pages.py tests/unit/test_auto_labeled_scraper_wikipedia.py
git commit -m "feat(auto-labeled): scraper CLI with Wikipedia source"
```

---

## Task 4: Scraper — Wikisource source

**Files:**
- Modify: `scripts/data/scrape_external_pages.py`
- Modify: `tests/unit/test_auto_labeled_scraper_wikipedia.py` (rename to cover Wikisource too, or add new test file)

Wikisource reuses the same Playwright screenshot logic; only the URL discovery differs. We extend the existing scraper file and add a test that exercises the Wikisource dispatch path.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_auto_labeled_scraper_wikisource.py`:

```python
from __future__ import annotations

from pathlib import Path

from scripts.data.auto_labeled_common import open_work_db
from scripts.data.scrape_external_pages import scrape_wikisource


def test_scrape_wikisource_inserts_pending_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "work.db"
    raw_dir = tmp_path / "raw" / "wikisource"
    raw_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._random_wikisource_page_urls",
        lambda limit: ["https://hi.wikisource.org/wiki/पृष्ठ:A/1"][:limit],
    )
    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._screenshot_url",
        lambda url, out_path: out_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100),
    )
    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._is_non_blank_png",
        lambda path: True,
    )

    conn = open_work_db(db_path)
    scrape_wikisource(conn=conn, limit=1, raw_dir=raw_dir)
    conn.commit()

    row = conn.execute("SELECT source, status FROM pages").fetchone()
    assert row == ("wikisource", "pending")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_auto_labeled_scraper_wikisource.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add Wikisource source to the scraper**

Edit `scripts/data/scrape_external_pages.py` — add the following after the `scrape_wikipedia` function and update the CLI dispatch:

```python
# --- Wikisource -----------------------------------------------------------------

WIKISOURCE_API = "https://hi.wikisource.org/w/api.php"
WIKISOURCE_BASE = "https://hi.wikisource.org/wiki/"


def _random_wikisource_page_urls(limit: int) -> list[str]:
    """Use the MediaWiki API on hi.wikisource to sample random pages in the Page: (ProofreadPage) namespace."""
    with httpx.Client(timeout=30.0) as client:
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnnamespace": 104,  # Page namespace for ProofreadPage
            "rnlimit": min(limit, 10),
        }
        urls: list[str] = []
        while len(urls) < limit:
            resp = client.get(WIKISOURCE_API, params=params)
            resp.raise_for_status()
            data = resp.json()
            for item in data["query"]["random"]:
                title = item["title"].replace(" ", "_")
                urls.append(WIKISOURCE_BASE + title)
                if len(urls) >= limit:
                    break
    return urls[:limit]


def scrape_wikisource(conn: sqlite3.Connection, limit: int, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    urls = _random_wikisource_page_urls(limit)
    for url in urls:
        out = raw_dir / f"{uuid.uuid4().hex}.png"
        try:
            _screenshot_url(url, out)
            if not _is_non_blank_png(out):
                _insert_page(conn, "wikisource", url, out, "fetch_failed", error="blank_render")
                continue
            _insert_page(conn, "wikisource", url, out, "pending")
        except Exception as exc:
            _insert_page(conn, "wikisource", url, out, "fetch_failed", error=str(exc)[:500])
```

Then update `main()` to dispatch:

```python
        if args.source == "wikipedia":
            scrape_wikipedia(conn=conn, limit=args.limit, raw_dir=raw_dir)
        elif args.source == "wikisource":
            scrape_wikisource(conn=conn, limit=args.limit, raw_dir=raw_dir)
        else:
            print(f"source {args.source} not yet implemented", file=sys.stderr)
            return 2
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_auto_labeled_scraper_wikisource.py tests/unit/test_auto_labeled_scraper_wikipedia.py -v`
Expected: All tests PASS (the Wikipedia tests from Task 3 still pass).

- [ ] **Step 5: Commit**

```bash
git add scripts/data/scrape_external_pages.py tests/unit/test_auto_labeled_scraper_wikisource.py
git commit -m "feat(auto-labeled): scraper Wikisource source"
```

---

## Task 5: Scraper — gov_pdf source

**Files:**
- Modify: `scripts/data/scrape_external_pages.py`
- Create: `scripts/data/auto_labeled_gov_seeds.txt` (small committed seed list)
- Create: `tests/unit/test_auto_labeled_scraper_gov_pdf.py`

Gov PDFs are downloaded with `httpx` and rasterized with `pdf2image`. The seed URL list is committed to the repo so runs are reproducible. Start the seed list with 10 well-known public-domain URLs (budget docs, egazette, court judgments) — when running the real pipeline you'll grow the list.

- [ ] **Step 1: Create the seed file**

Create `scripts/data/auto_labeled_gov_seeds.txt` with one URL per line. Start with a small list of 10 known public-domain Hindi PDF URLs. Example seed content (you will expand this list manually before running a full pipeline):

```
https://www.indiabudget.gov.in/doc/Budget_Speech.pdf
https://egazette.nic.in/WriteReadData/2024/ExampleHindi.pdf
https://main.sci.gov.in/supremecourt/2023/Example/hindi.pdf
```

Commit this list with a note that it is a placeholder starter list — the implementation plan's §13 open items in the spec flag the need to grow it.

- [ ] **Step 2: Write the failing test**

Create `tests/unit/test_auto_labeled_scraper_gov_pdf.py`:

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from scripts.data.auto_labeled_common import open_work_db
from scripts.data.scrape_external_pages import scrape_gov_pdf


def test_scrape_gov_pdf_rasterizes_and_inserts(tmp_path, monkeypatch):
    db_path = tmp_path / "work.db"
    raw_dir = tmp_path / "raw" / "gov_pdf"
    raw_dir.mkdir(parents=True)

    # Provide an in-memory seed list (one fake URL).
    seeds = tmp_path / "seeds.txt"
    seeds.write_text("https://example.gov.in/test.pdf\n", encoding="utf-8")

    # Stub downloader to write fake PDF bytes.
    def fake_download(url, dest):
        Path(dest).write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._download_pdf", fake_download
    )

    # Stub rasterizer to yield two PIL images.
    def fake_rasterize(pdf_path):
        return [Image.new("RGB", (800, 1000), "white"), Image.new("RGB", (800, 1000), "white")]

    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._rasterize_pdf", fake_rasterize
    )
    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._is_non_blank_png", lambda path: True
    )

    conn = open_work_db(db_path)
    scrape_gov_pdf(conn=conn, limit=2, raw_dir=raw_dir, seeds_path=seeds)
    conn.commit()

    rows = list(conn.execute("SELECT source, status FROM pages"))
    assert len(rows) == 2
    assert all(r == ("gov_pdf", "pending") for r in rows)
```

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/unit/test_auto_labeled_scraper_gov_pdf.py -v`
Expected: FAIL — `scrape_gov_pdf` not defined.

- [ ] **Step 4: Implement the gov_pdf source**

Add to `scripts/data/scrape_external_pages.py`:

```python
# --- Gov PDFs -------------------------------------------------------------------

DEFAULT_GOV_SEEDS = Path(__file__).parent / "auto_labeled_gov_seeds.txt"


def _load_seeds(path: Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _download_pdf(url: str, dest: Path) -> None:
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
    Path(dest).write_bytes(resp.content)


def _rasterize_pdf(pdf_path: Path) -> list:
    from pdf2image import convert_from_path
    return convert_from_path(str(pdf_path), dpi=300)


def scrape_gov_pdf(
    conn: sqlite3.Connection,
    limit: int,
    raw_dir: Path,
    seeds_path: Path = DEFAULT_GOV_SEEDS,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    seeds = _load_seeds(seeds_path)
    if not seeds:
        return
    pages_written = 0
    for url in seeds:
        if pages_written >= limit:
            break
        pdf_local = raw_dir / f"{uuid.uuid4().hex}.pdf"
        try:
            _download_pdf(url, pdf_local)
            images = _rasterize_pdf(pdf_local)
        except Exception as exc:
            _insert_page(conn, "gov_pdf", url, pdf_local, "fetch_failed", error=str(exc)[:500])
            continue
        for img in images:
            if pages_written >= limit:
                break
            out = raw_dir / f"{uuid.uuid4().hex}.png"
            img.save(out, format="PNG")
            if not _is_non_blank_png(out):
                _insert_page(conn, "gov_pdf", url, out, "fetch_failed", error="blank_render")
                continue
            _insert_page(conn, "gov_pdf", url, out, "pending")
            pages_written += 1
```

Update `main()`:

```python
        elif args.source == "gov_pdf":
            scrape_gov_pdf(conn=conn, limit=args.limit, raw_dir=raw_dir)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_auto_labeled_scraper_gov_pdf.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/data/scrape_external_pages.py scripts/data/auto_labeled_gov_seeds.txt tests/unit/test_auto_labeled_scraper_gov_pdf.py
git commit -m "feat(auto-labeled): scraper gov_pdf source with seed list"
```

---

## Task 6: Scraper — archive.org source

**Files:**
- Modify: `scripts/data/scrape_external_pages.py`
- Create: `tests/unit/test_auto_labeled_scraper_archive_org.py`

Archive.org uses the public Advanced Search API filtered to Hindi (`language:hin`) and text items. For each returned item, download the `_jp2.zip`, extract N pages, rasterize. Since the full download flow is heavy, the scraper uses a hard per-item byte cap and skips anything larger to prevent wedging.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_auto_labeled_scraper_archive_org.py`:

```python
from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.data.auto_labeled_common import open_work_db
from scripts.data.scrape_external_pages import scrape_archive_org


def test_scrape_archive_org_inserts_pending(tmp_path, monkeypatch):
    db_path = tmp_path / "work.db"
    raw_dir = tmp_path / "raw" / "archive_org"
    raw_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._search_archive_hindi_text_items",
        lambda limit_items: [("item-001", "https://archive.org/download/item-001/item-001_jp2.zip")],
    )

    def fake_fetch(url, max_pages):
        return [Image.new("L", (900, 1200), 255) for _ in range(max_pages)]

    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._fetch_and_extract_jp2_pages", fake_fetch
    )
    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._is_non_blank_png", lambda path: True
    )

    conn = open_work_db(db_path)
    scrape_archive_org(conn=conn, limit=3, raw_dir=raw_dir)
    conn.commit()

    rows = list(conn.execute("SELECT source, status FROM pages"))
    assert len(rows) == 3
    assert all(r == ("archive_org", "pending") for r in rows)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_auto_labeled_scraper_archive_org.py -v`
Expected: FAIL — function missing.

- [ ] **Step 3: Implement archive.org source**

Add to `scripts/data/scrape_external_pages.py`:

```python
# --- archive.org ---------------------------------------------------------------

ARCHIVE_SEARCH_URL = "https://archive.org/advancedsearch.php"
ARCHIVE_MAX_ITEM_BYTES = 300 * 1024 * 1024  # 300 MB cap per item


def _search_archive_hindi_text_items(limit_items: int) -> list[tuple[str, str]]:
    """Return list of (identifier, jp2_zip_url) for Hindi text items."""
    params = {
        "q": "language:hin AND mediatype:texts",
        "fl[]": ["identifier"],
        "rows": limit_items,
        "page": 1,
        "output": "json",
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(ARCHIVE_SEARCH_URL, params=params)
        resp.raise_for_status()
        docs = resp.json().get("response", {}).get("docs", [])
    out: list[tuple[str, str]] = []
    for d in docs:
        ident = d["identifier"]
        jp2 = f"https://archive.org/download/{ident}/{ident}_jp2.zip"
        out.append((ident, jp2))
    return out


def _fetch_and_extract_jp2_pages(url: str, max_pages: int) -> list:
    """Download a _jp2.zip and return up to max_pages PIL images."""
    import io
    import zipfile
    from PIL import Image

    with httpx.Client(timeout=600.0, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            size = int(resp.headers.get("content-length", 0))
            if size and size > ARCHIVE_MAX_ITEM_BYTES:
                raise RuntimeError(f"item too large: {size} bytes")
            buf = io.BytesIO()
            for chunk in resp.iter_bytes():
                buf.write(chunk)
                if buf.tell() > ARCHIVE_MAX_ITEM_BYTES:
                    raise RuntimeError("item exceeded cap during stream")
    buf.seek(0)
    images: list = []
    with zipfile.ZipFile(buf) as zf:
        jp2_names = sorted(n for n in zf.namelist() if n.endswith(".jp2"))[:max_pages]
        for name in jp2_names:
            with zf.open(name) as jf:
                images.append(Image.open(io.BytesIO(jf.read())).convert("L"))
    return images


def scrape_archive_org(conn: sqlite3.Connection, limit: int, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    items_needed = max(1, limit // 20)  # ~20 pages per item is a conservative estimate
    items = _search_archive_hindi_text_items(items_needed + 5)
    pages_written = 0
    for ident, url in items:
        if pages_written >= limit:
            break
        try:
            imgs = _fetch_and_extract_jp2_pages(url, max_pages=min(20, limit - pages_written))
        except Exception as exc:
            _insert_page(conn, "archive_org", url, raw_dir / f"{ident}.skip", "fetch_failed", error=str(exc)[:500])
            continue
        for img in imgs:
            if pages_written >= limit:
                break
            out = raw_dir / f"{uuid.uuid4().hex}.png"
            img.save(out, format="PNG")
            if not _is_non_blank_png(out):
                _insert_page(conn, "archive_org", url, out, "fetch_failed", error="blank_render")
                continue
            _insert_page(conn, "archive_org", url, out, "pending")
            pages_written += 1
```

Update `main()`:

```python
        elif args.source == "archive_org":
            scrape_archive_org(conn=conn, limit=args.limit, raw_dir=raw_dir)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_auto_labeled_scraper_archive_org.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/data/scrape_external_pages.py tests/unit/test_auto_labeled_scraper_archive_org.py
git commit -m "feat(auto-labeled): scraper archive.org source"
```

---

## Task 7: Labeler CLI with async drain

**Files:**
- Create: `scripts/data/label_with_cloud_ocr.py`
- Create: `tests/unit/test_auto_labeled_labeler.py`

The labeler drains `pages` rows with `status='pending'`, labels each through the provider interface, writes `words` rows, and transitions pages to `labeled` or `label_failed`. Concurrency is via `asyncio.Semaphore` and retries via `tenacity`. For tests, the fake provider is synchronous — the labeler wraps it in `asyncio.to_thread`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_auto_labeled_labeler.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.data.auto_labeled_common import open_work_db
from scripts.data.auto_labeled_provider import FakeProvider
from scripts.data.label_with_cloud_ocr import run_labeler

FAKE_JSON = Path(__file__).parent.parent / "fixtures" / "auto_labeled" / "fake_responses.json"


def _seed_pending_page(conn, local_path: Path, source="wikipedia"):
    conn.execute(
        "INSERT INTO pages (source, uri, local_path, fetched_at, status) VALUES (?,?,?,?,?)",
        (source, "https://example/", str(local_path), "2026-04-16T00:00:00Z", "pending"),
    )
    conn.commit()


def test_labeler_drains_pending_and_writes_words(tmp_path):
    db_path = tmp_path / "work.db"
    conn = open_work_db(db_path)
    img_path = tmp_path / "page_wiki_001.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    _seed_pending_page(conn, img_path)
    conn.close()

    provider = FakeProvider(responses_path=FAKE_JSON)
    run_labeler(db_path=db_path, provider=provider, concurrency=2)

    conn = open_work_db(db_path)
    page_status = conn.execute("SELECT status FROM pages").fetchone()[0]
    assert page_status == "labeled"
    word_rows = list(conn.execute("SELECT text_nfc, conf, status FROM words"))
    assert len(word_rows) == 3
    assert word_rows[0][0] == "भारत"
    assert word_rows[0][2] == "labeled"


def test_labeler_handles_empty_response(tmp_path):
    db_path = tmp_path / "work.db"
    conn = open_work_db(db_path)
    img_path = tmp_path / "page_empty.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    _seed_pending_page(conn, img_path)
    conn.close()

    provider = FakeProvider(responses_path=FAKE_JSON)
    run_labeler(db_path=db_path, provider=provider, concurrency=1)

    conn = open_work_db(db_path)
    assert conn.execute("SELECT status FROM pages").fetchone()[0] == "labeled"
    assert conn.execute("SELECT COUNT(*) FROM words").fetchone()[0] == 0


def test_labeler_marks_failure_on_provider_exception(tmp_path):
    db_path = tmp_path / "work.db"
    conn = open_work_db(db_path)
    img_path = tmp_path / "does_not_exist.png"
    _seed_pending_page(conn, img_path)
    conn.close()

    class AngryProvider:
        name = "angry"
        def label_page(self, path):
            raise RuntimeError("boom")

    run_labeler(db_path=db_path, provider=AngryProvider(), concurrency=1, max_attempts=1)

    conn = open_work_db(db_path)
    row = conn.execute("SELECT status, error FROM pages").fetchone()
    assert row[0] == "label_failed"
    assert "boom" in (row[1] or "")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_auto_labeled_labeler.py -v`
Expected: FAIL — `run_labeler` not defined.

- [ ] **Step 3: Implement the labeler**

Create `scripts/data/label_with_cloud_ocr.py`:

```python
from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path

from scripts.data.auto_labeled_common import normalize_text, open_work_db
from scripts.data.auto_labeled_provider import Provider, WordBox, get_provider


async def _label_one(
    provider: Provider,
    page_id: int,
    local_path: Path,
    max_attempts: int,
) -> tuple[int, list[WordBox] | None, str | None]:
    last_exc: str | None = None
    for attempt in range(max_attempts):
        try:
            boxes = await asyncio.to_thread(provider.label_page, local_path)
            return page_id, boxes, None
        except Exception as exc:
            last_exc = str(exc)[:500]
            if attempt + 1 < max_attempts:
                await asyncio.sleep(min(2 ** attempt, 30))
    return page_id, None, last_exc


async def _drain(
    pending: list[tuple[int, str]],
    provider: Provider,
    concurrency: int,
    max_attempts: int,
) -> list[tuple[int, list[WordBox] | None, str | None]]:
    sem = asyncio.Semaphore(concurrency)

    async def _guarded(page_id: int, local_path: str):
        async with sem:
            return await _label_one(provider, page_id, Path(local_path), max_attempts)

    return await asyncio.gather(*(_guarded(pid, p) for pid, p in pending))


def run_labeler(
    db_path: Path,
    provider: Provider,
    concurrency: int = 8,
    max_attempts: int = 3,
) -> None:
    conn = open_work_db(db_path)
    pending = list(
        conn.execute("SELECT id, local_path FROM pages WHERE status = 'pending'")
    )
    if not pending:
        conn.close()
        return
    results = asyncio.run(_drain(pending, provider, concurrency, max_attempts))

    for page_id, boxes, err in results:
        if err is not None or boxes is None:
            conn.execute(
                "UPDATE pages SET status='label_failed', error=? WHERE id=?",
                (err, page_id),
            )
            continue
        for b in boxes:
            text = normalize_text(b.text)
            if not text:
                continue
            conn.execute(
                "INSERT INTO words (page_id, x, y, w, h, text_nfc, conf, provider, status)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'labeled')",
                (page_id, b.x, b.y, b.w, b.h, text, b.conf, provider.name),
            )
        conn.execute("UPDATE pages SET status='labeled' WHERE id=?", (page_id,))
    conn.commit()
    conn.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Label pending pages via a cloud OCR provider.")
    p.add_argument("--db", type=Path, default=Path("data/external/auto_labeled/work.db"))
    p.add_argument("--provider", choices=("azure", "fake"), default="azure")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--fake-responses", type=Path, help="JSON fixture for --provider fake")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    provider = get_provider(
        args.provider,
        fake_responses_path=args.fake_responses,
    )
    run_labeler(db_path=args.db, provider=provider, concurrency=args.concurrency)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_auto_labeled_labeler.py -v`
Expected: All 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/data/label_with_cloud_ocr.py tests/unit/test_auto_labeled_labeler.py
git commit -m "feat(auto-labeled): labeler CLI with async drain and retry"
```

---

## Task 8: Filter stage — confidence, OOV, bbox, dup

**Files:**
- Create: `scripts/data/filter_and_crop_words.py`
- Create: `tests/unit/test_auto_labeled_filter_basics.py`

This task establishes the filter CLI skeleton and the first four filter stages (confidence, OOV, bbox, dup). Text-disjoint filtering comes in Task 9, QA candidate selection in Task 10. The filter CLI file will grow across tasks 8–12.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_auto_labeled_filter_basics.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from scripts.data.auto_labeled_common import open_work_db
from scripts.data.filter_and_crop_words import (
    filter_bbox,
    filter_confidence,
    filter_dup,
    filter_oov,
)


def _seed_page(conn, page_id: int, w=1000, h=1000):
    conn.execute(
        "INSERT INTO pages (id, source, uri, local_path, fetched_at, status)"
        " VALUES (?, 'wikipedia', 'u', 'p.png', '2026-04-16', 'labeled')",
        (page_id,),
    )


def _seed_word(conn, page_id, text, conf, x=0, y=0, w=10, h=10, status="labeled"):
    conn.execute(
        "INSERT INTO words (page_id, x, y, w, h, text_nfc, conf, provider, status)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (page_id, x, y, w, h, text, conf, "fake", status),
    )


def test_filter_confidence_below_threshold(tmp_path):
    conn = open_work_db(tmp_path / "w.db")
    _seed_page(conn, 1)
    _seed_word(conn, 1, "भारत", 0.90)
    _seed_word(conn, 1, "कम", 0.50)
    conn.commit()
    filter_confidence(conn, min_conf=0.85)
    statuses = sorted(row[0] for row in conn.execute("SELECT status FROM words"))
    assert statuses == ["filtered_low_conf", "labeled"]


def test_filter_oov(tmp_path):
    conn = open_work_db(tmp_path / "w.db")
    _seed_page(conn, 1)
    _seed_word(conn, 1, "भारत", 0.99)
    _seed_word(conn, 1, "ZZZ$", 0.99)
    conn.commit()
    vocab = {"भ", "ा", "र", "त"}
    filter_oov(conn, vocab=vocab)
    rows = dict(conn.execute("SELECT text_nfc, status FROM words"))
    assert rows["भारत"] == "labeled"
    assert rows["ZZZ$"] == "filtered_oov"


def test_filter_bbox(tmp_path):
    conn = open_work_db(tmp_path / "w.db")
    _seed_page(conn, 1)
    _seed_word(conn, 1, "A", 0.99, x=10, y=10, w=50, h=20)  # valid inside 1000x1000
    _seed_word(conn, 1, "B", 0.99, x=990, y=990, w=50, h=50)  # extends past bounds
    conn.commit()
    # page size lookup uses local_path image dims; test provides fixed (w=100,h=100)
    filter_bbox(conn, page_dims={1: (100, 100)})
    rows = dict(conn.execute("SELECT text_nfc, status FROM words"))
    assert rows["A"] == "labeled"
    assert rows["B"] == "filtered_invalid_bbox"


def test_filter_dup(tmp_path):
    conn = open_work_db(tmp_path / "w.db")
    _seed_page(conn, 1)
    _seed_word(conn, 1, "भारत", 0.99, x=10, y=10)
    _seed_word(conn, 1, "भारत", 0.99, x=10, y=10)  # exact dup
    _seed_word(conn, 1, "भारत", 0.99, x=50, y=10)  # different x
    conn.commit()
    filter_dup(conn)
    statuses = sorted(row[0] for row in conn.execute("SELECT status FROM words ORDER BY id"))
    # Exactly one should be filtered_dup; two remain labeled.
    assert statuses.count("labeled") == 2
    assert statuses.count("filtered_dup") == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_auto_labeled_filter_basics.py -v`
Expected: FAIL — filter functions not defined.

- [ ] **Step 3: Implement filter basics**

Create `scripts/data/filter_and_crop_words.py`:

```python
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from PIL import Image

from scripts.data.auto_labeled_common import open_work_db


def filter_confidence(conn: sqlite3.Connection, min_conf: float) -> None:
    conn.execute(
        "UPDATE words SET status='filtered_low_conf', filter_reason=?"
        " WHERE status='labeled' AND conf < ?",
        (f"conf<{min_conf}", min_conf),
    )


def filter_oov(conn: sqlite3.Connection, vocab: set[str]) -> None:
    rows = list(conn.execute("SELECT id, text_nfc FROM words WHERE status='labeled'"))
    bad_ids: list[int] = []
    for wid, text in rows:
        if any(ch not in vocab for ch in text):
            bad_ids.append(wid)
    if bad_ids:
        conn.executemany(
            "UPDATE words SET status='filtered_oov', filter_reason='oov_char' WHERE id=?",
            [(i,) for i in bad_ids],
        )


def filter_bbox(
    conn: sqlite3.Connection,
    page_dims: dict[int, tuple[int, int]] | None = None,
) -> None:
    if page_dims is None:
        page_dims = {}
        for pid, lp in conn.execute("SELECT id, local_path FROM pages"):
            try:
                with Image.open(lp) as im:
                    page_dims[pid] = im.size
            except Exception:
                page_dims[pid] = (0, 0)

    bad_ids: list[int] = []
    for wid, pid, x, y, w, h in conn.execute(
        "SELECT id, page_id, x, y, w, h FROM words WHERE status='labeled'"
    ):
        W, H = page_dims.get(pid, (0, 0))
        x2 = min(x + w, W)
        y2 = min(y + h, H)
        x = max(x, 0)
        y = max(y, 0)
        if x2 - x <= 0 or y2 - y <= 0 or W == 0 or H == 0:
            bad_ids.append(wid)
    if bad_ids:
        conn.executemany(
            "UPDATE words SET status='filtered_invalid_bbox', filter_reason='bbox_oob' WHERE id=?",
            [(i,) for i in bad_ids],
        )


def filter_dup(conn: sqlite3.Connection) -> None:
    """Keep the earliest (lowest id) row for each (page_id, text_nfc, x, y) tuple; mark the rest as dup."""
    dup_ids: list[int] = []
    seen: set[tuple[int, str, int, int]] = set()
    for wid, pid, text, x, y in conn.execute(
        "SELECT id, page_id, text_nfc, x, y FROM words WHERE status='labeled' ORDER BY id"
    ):
        key = (pid, text, x, y)
        if key in seen:
            dup_ids.append(wid)
        else:
            seen.add(key)
    if dup_ids:
        conn.executemany(
            "UPDATE words SET status='filtered_dup', filter_reason='dup' WHERE id=?",
            [(i,) for i in dup_ids],
        )


def load_vocab(vocab_path: Path) -> set[str]:
    data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return set(data)
    if isinstance(data, dict):
        return set(data.keys())
    raise ValueError(f"unsupported vocab format in {vocab_path}")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_auto_labeled_filter_basics.py -v`
Expected: All 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/data/filter_and_crop_words.py tests/unit/test_auto_labeled_filter_basics.py
git commit -m "feat(auto-labeled): filter stages for confidence, OOV, bbox, dup"
```

---

## Task 9: Filter — text-disjoint policy B rule 1

**Files:**
- Modify: `scripts/data/filter_and_crop_words.py`
- Create: `tests/unit/test_auto_labeled_filter_disjoint.py`

Rule 1: no `labeled` row's text may appear in the existing val/test/benchmark strings. Implemented as a single `UPDATE ... WHERE text_nfc IN (...)` using temp tables loaded from flat files.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_auto_labeled_filter_disjoint.py`:

```python
from __future__ import annotations

from pathlib import Path

from scripts.data.auto_labeled_common import open_work_db
from scripts.data.filter_and_crop_words import apply_text_disjoint_rule_1


def _seed_page(conn, page_id=1):
    conn.execute(
        "INSERT INTO pages (id, source, uri, local_path, fetched_at, status)"
        " VALUES (?, 'wikipedia', 'u', 'p.png', '2026-04-16', 'labeled')",
        (page_id,),
    )


def _seed_word(conn, text, page_id=1):
    conn.execute(
        "INSERT INTO words (page_id, x, y, w, h, text_nfc, conf, provider, status)"
        " VALUES (?,0,0,10,10,?,0.99,'fake','labeled')",
        (page_id, text),
    )


def test_rule_1_filters_matching_text(tmp_path):
    conn = open_work_db(tmp_path / "w.db")
    _seed_page(conn)
    for t in ["भारत", "हिंदी", "नया"]:
        _seed_word(conn, t)
    conn.commit()

    val = tmp_path / "val.txt"
    val.write_text("/a.png|भारत\n", encoding="utf-8")
    test = tmp_path / "test.txt"
    test.write_text("/a.png|हिंदी\n", encoding="utf-8")

    apply_text_disjoint_rule_1(
        conn,
        existing_val=[val],
        existing_test=[test],
        existing_benchmarks=[],
    )
    rows = dict(conn.execute("SELECT text_nfc, status FROM words"))
    assert rows["भारत"] == "filtered_text_leak"
    assert rows["हिंदी"] == "filtered_text_leak"
    assert rows["नया"] == "labeled"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_auto_labeled_filter_disjoint.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement rule 1**

Append to `scripts/data/filter_and_crop_words.py`:

```python
from scripts.data.auto_labeled_common import load_text_strings


def _load_all_strings(val, test, bench) -> tuple[set[str], set[str], set[str]]:
    return (
        load_text_strings(val),
        load_text_strings(test),
        load_text_strings(bench),
    )


def apply_text_disjoint_rule_1(
    conn: sqlite3.Connection,
    existing_val: list[Path],
    existing_test: list[Path],
    existing_benchmarks: list[Path],
) -> None:
    val, test, bench = _load_all_strings(existing_val, existing_test, existing_benchmarks)

    conn.execute("DROP TABLE IF EXISTS temp.existing_val_strings")
    conn.execute("DROP TABLE IF EXISTS temp.existing_test_strings")
    conn.execute("DROP TABLE IF EXISTS temp.existing_bench_strings")
    conn.execute("CREATE TEMP TABLE existing_val_strings (text_nfc TEXT PRIMARY KEY)")
    conn.execute("CREATE TEMP TABLE existing_test_strings (text_nfc TEXT PRIMARY KEY)")
    conn.execute("CREATE TEMP TABLE existing_bench_strings (text_nfc TEXT PRIMARY KEY)")
    conn.executemany("INSERT OR IGNORE INTO existing_val_strings VALUES (?)", [(s,) for s in val])
    conn.executemany("INSERT OR IGNORE INTO existing_test_strings VALUES (?)", [(s,) for s in test])
    conn.executemany("INSERT OR IGNORE INTO existing_bench_strings VALUES (?)", [(s,) for s in bench])

    conn.execute(
        """
        UPDATE words
        SET status='filtered_text_leak',
            filter_reason='leak_vs_existing_val_test_bench'
        WHERE status='labeled'
          AND text_nfc IN (
              SELECT text_nfc FROM existing_val_strings
              UNION
              SELECT text_nfc FROM existing_test_strings
              UNION
              SELECT text_nfc FROM existing_bench_strings
          )
        """
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_auto_labeled_filter_disjoint.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/data/filter_and_crop_words.py tests/unit/test_auto_labeled_filter_disjoint.py
git commit -m "feat(auto-labeled): filter policy B rule 1 (leak vs existing val/test/benchmarks)"
```

---

## Task 10: Filter — promote kept rows and select QA candidates

**Files:**
- Modify: `scripts/data/filter_and_crop_words.py`
- Create: `tests/unit/test_auto_labeled_filter_qa_selection.py`

This task implements the `labeled → kept` promotion and the unique-text QA candidate selection from spec §5.3. The CTE in the spec relies on SQLite window functions; the test exercises them against in-memory DBs.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_auto_labeled_filter_qa_selection.py`:

```python
from __future__ import annotations

from pathlib import Path

from scripts.data.auto_labeled_common import open_work_db
from scripts.data.filter_and_crop_words import (
    promote_labeled_to_kept,
    select_qa_candidates,
    symmetric_qa_eviction,
)


def _seed(conn, rows):
    conn.execute(
        "INSERT INTO pages (id, source, uri, local_path, fetched_at, status)"
        " VALUES (1, 'wikipedia', 'u', 'p.png', '2026-04-16', 'labeled')"
    )
    conn.execute(
        "INSERT INTO pages (id, source, uri, local_path, fetched_at, status)"
        " VALUES (2, 'archive_org', 'u', 'p2.png', '2026-04-16', 'labeled')"
    )
    for page_id, text, status in rows:
        conn.execute(
            "INSERT INTO words (page_id, x, y, w, h, text_nfc, conf, provider, status)"
            " VALUES (?,0,0,10,10,?,0.99,'fake',?)",
            (page_id, text, status),
        )


def test_promote_labeled_to_kept(tmp_path):
    conn = open_work_db(tmp_path / "w.db")
    _seed(conn, [(1, "A", "labeled"), (1, "B", "filtered_low_conf")])
    conn.commit()
    promote_labeled_to_kept(conn)
    rows = dict(conn.execute("SELECT text_nfc, status FROM words"))
    assert rows["A"] == "kept"
    assert rows["B"] == "filtered_low_conf"


def test_select_qa_candidates_picks_unique_text(tmp_path):
    conn = open_work_db(tmp_path / "w.db")
    # 3 unique-text rows on page 1 (wikipedia), 2 unique on page 2 (archive_org).
    _seed(
        conn,
        [
            (1, "uniq_w_1", "kept"),
            (1, "uniq_w_2", "kept"),
            (1, "uniq_w_3", "kept"),
            (1, "dup_text", "kept"),  # appears twice -> excluded
            (1, "dup_text", "kept"),
            (2, "uniq_a_1", "kept"),
            (2, "uniq_a_2", "kept"),
        ],
    )
    conn.commit()

    select_qa_candidates(conn, target_per_source=2)

    candidates = [
        (src, text)
        for src, text in conn.execute(
            "SELECT p.source, w.text_nfc FROM words w JOIN pages p ON p.id = w.page_id"
            " WHERE w.status='exported_qa_candidate' ORDER BY w.id"
        )
    ]
    # Each source got exactly 2, all unique-text (no 'dup_text').
    by_source = {}
    for src, text in candidates:
        by_source.setdefault(src, []).append(text)
    assert len(by_source["wikipedia"]) == 2
    assert len(by_source["archive_org"]) == 2
    assert "dup_text" not in {t for _, t in candidates}


def test_symmetric_eviction_is_noop_after_unique_selection(tmp_path):
    conn = open_work_db(tmp_path / "w.db")
    _seed(conn, [(1, "alpha", "kept"), (1, "beta", "kept")])
    conn.commit()
    select_qa_candidates(conn, target_per_source=1)
    affected = symmetric_qa_eviction(conn)
    assert affected == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_auto_labeled_filter_qa_selection.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement kept-promotion, QA selection, symmetric eviction**

Append to `scripts/data/filter_and_crop_words.py`:

```python
def promote_labeled_to_kept(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE words SET status='kept' WHERE status='labeled'")


def select_qa_candidates(conn: sqlite3.Connection, target_per_source: int = 500) -> None:
    """Promote kept rows to exported_qa_candidate using unique-text predicate.

    Assumes existing_val_strings, existing_test_strings, existing_bench_strings temp
    tables exist from apply_text_disjoint_rule_1; if not, creates empty ones so the
    query still works.
    """
    for name in ("existing_val_strings", "existing_test_strings", "existing_bench_strings"):
        conn.execute(f"CREATE TEMP TABLE IF NOT EXISTS {name} (text_nfc TEXT PRIMARY KEY)")

    conn.execute(
        """
        WITH unique_text_kept AS (
            SELECT id, text_nfc, page_id
            FROM words
            WHERE status = 'kept'
              AND text_nfc IN (
                  SELECT text_nfc FROM words WHERE status = 'kept'
                  GROUP BY text_nfc HAVING COUNT(*) = 1
              )
              AND text_nfc NOT IN (SELECT text_nfc FROM existing_val_strings)
              AND text_nfc NOT IN (SELECT text_nfc FROM existing_test_strings)
              AND text_nfc NOT IN (SELECT text_nfc FROM existing_bench_strings)
        ),
        ranked AS (
            SELECT u.id, u.text_nfc, p.source,
                   ROW_NUMBER() OVER (PARTITION BY p.source ORDER BY RANDOM()) AS rn
            FROM unique_text_kept u JOIN pages p ON p.id = u.page_id
        )
        UPDATE words
        SET status='exported_qa_candidate'
        WHERE id IN (SELECT id FROM ranked WHERE rn <= ?)
        """,
        (target_per_source,),
    )


def symmetric_qa_eviction(conn: sqlite3.Connection) -> int:
    """Belt-and-suspenders: evict any kept row whose text matches an exported candidate.

    Returns the number of rows affected. In a correct run this must be zero.
    """
    cur = conn.execute(
        """
        UPDATE words
        SET status='filtered_text_leak',
            filter_reason='leak_vs_qa_candidate'
        WHERE status='kept'
          AND text_nfc IN (
              SELECT DISTINCT text_nfc FROM words WHERE status='exported_qa_candidate'
          )
        """
    )
    return cur.rowcount
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_auto_labeled_filter_qa_selection.py -v`
Expected: All 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/data/filter_and_crop_words.py tests/unit/test_auto_labeled_filter_qa_selection.py
git commit -m "feat(auto-labeled): kept promotion, unique-text QA selection, symmetric eviction"
```

---

## Task 11: Filter — audits (count-based and Levenshtein random sample)

**Files:**
- Modify: `scripts/data/filter_and_crop_words.py`
- Create: `tests/unit/test_auto_labeled_filter_audits.py`

The audits are a correctness gate. If any count-based audit is nonzero, the filter stage must abort with a loud error. The Levenshtein sample is written to a manifest file for human spot-check.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_auto_labeled_filter_audits.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.data.auto_labeled_common import open_work_db
from scripts.data.filter_and_crop_words import (
    apply_text_disjoint_rule_1,
    audit_disjointness,
    levenshtein_audit_sample,
    promote_labeled_to_kept,
    select_qa_candidates,
)


def _seed(conn):
    conn.execute(
        "INSERT INTO pages (id, source, uri, local_path, fetched_at, status)"
        " VALUES (1, 'wikipedia', 'u', 'p.png', '2026-04-16', 'labeled')"
    )


def test_audit_clean_when_no_leakage(tmp_path):
    conn = open_work_db(tmp_path / "w.db")
    _seed(conn)
    conn.execute(
        "INSERT INTO words (page_id, x, y, w, h, text_nfc, conf, provider, status)"
        " VALUES (1,0,0,10,10,'unique_string',0.99,'fake','kept')"
    )
    conn.commit()
    apply_text_disjoint_rule_1(conn, [], [], [])
    counts = audit_disjointness(conn)
    assert all(v == 0 for v in counts.values())


def test_audit_detects_injected_leak(tmp_path):
    conn = open_work_db(tmp_path / "w.db")
    _seed(conn)
    conn.execute(
        "INSERT INTO words (page_id, x, y, w, h, text_nfc, conf, provider, status)"
        " VALUES (1,0,0,10,10,'bad_word',0.99,'fake','kept')"
    )
    conn.commit()
    # Load val file with 'bad_word' after the kept row is already there and WITHOUT
    # running rule 1 — the audit must catch it.
    val = tmp_path / "val.txt"
    val.write_text("/a.png|bad_word\n", encoding="utf-8")
    from scripts.data.auto_labeled_common import load_text_strings

    for name in ("existing_val_strings", "existing_test_strings", "existing_bench_strings"):
        conn.execute(f"CREATE TEMP TABLE IF NOT EXISTS {name} (text_nfc TEXT PRIMARY KEY)")
    conn.executemany(
        "INSERT OR IGNORE INTO existing_val_strings VALUES (?)",
        [(s,) for s in load_text_strings([val])],
    )
    counts = audit_disjointness(conn)
    assert counts["kept_vs_existing_val"] == 1


def test_levenshtein_audit_flags_near_duplicates(tmp_path):
    kept_texts = ["भारत"]
    val_texts = ["भारत "]  # trailing space — should be caught if someone forgot to strip
    pairs = levenshtein_audit_sample(kept_texts, val_texts, sample_size=10)
    assert len(pairs) >= 1
    assert any(p[0] == "भारत" for p in pairs)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_auto_labeled_filter_audits.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the audits**

Append to `scripts/data/filter_and_crop_words.py`:

```python
def audit_disjointness(conn: sqlite3.Connection) -> dict[str, int]:
    """Return the four audit counts. All must be zero in a correct run."""
    def _count(sql: str) -> int:
        return conn.execute(sql).fetchone()[0]

    return {
        "kept_vs_existing_val": _count(
            "SELECT COUNT(*) FROM words"
            " WHERE status='kept' AND text_nfc IN (SELECT text_nfc FROM existing_val_strings)"
        ),
        "kept_vs_existing_test": _count(
            "SELECT COUNT(*) FROM words"
            " WHERE status='kept' AND text_nfc IN (SELECT text_nfc FROM existing_test_strings)"
        ),
        "kept_vs_existing_bench": _count(
            "SELECT COUNT(*) FROM words"
            " WHERE status='kept' AND text_nfc IN (SELECT text_nfc FROM existing_bench_strings)"
        ),
        "kept_vs_qa_candidate": _count(
            "SELECT COUNT(*) FROM words w1"
            " WHERE w1.status='kept'"
            " AND w1.text_nfc IN (SELECT text_nfc FROM words WHERE status='exported_qa_candidate')"
        ),
    }


def levenshtein_audit_sample(
    kept_texts: list[str],
    val_texts: list[str],
    sample_size: int = 100,
) -> list[tuple[str, str, int]]:
    """Return up to sample_size (kept, val, distance) triples where Levenshtein <= 1."""
    import random
    try:
        from Levenshtein import distance as lev  # type: ignore
    except ImportError:
        def lev(a: str, b: str) -> int:  # fallback, slow
            if a == b:
                return 0
            return 2  # treats everything non-equal as distance 2 -> no matches

    out: list[tuple[str, str, int]] = []
    kept_sample = random.sample(kept_texts, min(len(kept_texts), 500))
    val_sample = random.sample(val_texts, min(len(val_texts), 500))
    for kt in kept_sample:
        for vt in val_sample:
            d = lev(kt, vt)
            if d <= 1:
                out.append((kt, vt, d))
                if len(out) >= sample_size:
                    return out
    return out
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_auto_labeled_filter_audits.py -v`
Expected: All 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/data/filter_and_crop_words.py tests/unit/test_auto_labeled_filter_audits.py
git commit -m "feat(auto-labeled): disjointness audits (count + Levenshtein sample)"
```

---

## Task 12: Filter — cropping, label emission, CLI entry point

**Files:**
- Modify: `scripts/data/filter_and_crop_words.py`
- Create: `tests/unit/test_auto_labeled_filter_emit.py`

Final step of the filter stage: crop kept word-boxes to PNG files, emit normalized label files, write the manifest, and expose the full stage as a CLI `main()`. Auto-val split is 10% stratified by source, implemented as a second `ROW_NUMBER() OVER (PARTITION BY ...)` pass.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_auto_labeled_filter_emit.py`:

```python
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from scripts.data.auto_labeled_common import open_work_db
from scripts.data.filter_and_crop_words import (
    emit_labels_and_crops,
    promote_labeled_to_kept,
)


def test_emit_writes_crops_and_label_files(tmp_path):
    page_image = tmp_path / "page.png"
    Image.new("L", (200, 100), 255).save(page_image)

    conn = open_work_db(tmp_path / "w.db")
    conn.execute(
        "INSERT INTO pages (id, source, uri, local_path, fetched_at, status)"
        " VALUES (1, 'wikipedia', 'u', ?, '2026-04-16', 'labeled')",
        (str(page_image),),
    )
    conn.execute(
        "INSERT INTO words (page_id, x, y, w, h, text_nfc, conf, provider, status)"
        " VALUES (1, 10, 20, 50, 30, 'alpha', 0.99, 'fake', 'kept')"
    )
    conn.execute(
        "INSERT INTO words (page_id, x, y, w, h, text_nfc, conf, provider, status)"
        " VALUES (1, 80, 20, 50, 30, 'beta', 0.99, 'fake', 'exported_qa_candidate')"
    )
    conn.commit()

    out_dir = tmp_path / "normalized"
    emit_labels_and_crops(
        conn,
        normalized_dir=out_dir,
        auto_val_fraction=0.0,  # no auto-val in this tiny test
    )

    train_file = out_dir / "train_labels.txt"
    qa_file = out_dir / "qa_candidates.txt"
    assert train_file.exists()
    assert qa_file.exists()

    train_lines = train_file.read_text(encoding="utf-8").strip().splitlines()
    qa_lines = qa_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(train_lines) == 1
    assert len(qa_lines) == 1
    assert train_lines[0].endswith("|alpha")
    assert qa_lines[0].endswith("|beta")

    crops = list((out_dir / "crops").glob("*.png"))
    assert len(crops) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_auto_labeled_filter_emit.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement emit + CLI**

Append to `scripts/data/filter_and_crop_words.py`:

```python
def _crop_word(page_path: Path, x: int, y: int, w: int, h: int, out_path: Path, target_h: int = 32) -> None:
    with Image.open(page_path) as im:
        W, H = im.size
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + w, W)
        y2 = min(y + h, H)
        crop = im.crop((x1, y1, x2, y2))
        if crop.height != target_h:
            ratio = target_h / max(crop.height, 1)
            new_w = max(1, int(crop.width * ratio))
            crop = crop.resize((new_w, target_h))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_path, format="PNG")


def emit_labels_and_crops(
    conn: sqlite3.Connection,
    normalized_dir: Path,
    auto_val_fraction: float = 0.10,
) -> dict[str, int]:
    crops_dir = normalized_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    page_paths = dict(conn.execute("SELECT id, local_path FROM pages"))

    # Training rows = status 'kept'. Auto-val is a random fraction stratified by source.
    kept_rows = list(
        conn.execute(
            "SELECT w.id, w.page_id, w.x, w.y, w.w, w.h, w.text_nfc, p.source"
            " FROM words w JOIN pages p ON p.id=w.page_id WHERE w.status='kept' ORDER BY w.id"
        )
    )
    qa_rows = list(
        conn.execute(
            "SELECT w.id, w.page_id, w.x, w.y, w.w, w.h, w.text_nfc, p.source"
            " FROM words w JOIN pages p ON p.id=w.page_id WHERE w.status='exported_qa_candidate' ORDER BY w.id"
        )
    )

    import random
    val_pick: set[int] = set()
    if auto_val_fraction > 0:
        by_source: dict[str, list[int]] = {}
        for wid, _, _, _, _, _, _, src in kept_rows:
            by_source.setdefault(src, []).append(wid)
        for src, ids in by_source.items():
            k = max(1, int(len(ids) * auto_val_fraction))
            val_pick.update(random.sample(ids, min(k, len(ids))))

    train_lines: list[str] = []
    val_lines: list[str] = []
    qa_lines: list[str] = []

    for wid, pid, x, y, w, h, text, _src in kept_rows:
        out = crops_dir / f"{wid}.png"
        _crop_word(Path(page_paths[pid]), x, y, w, h, out)
        line = f"{out}|{text}"
        if wid in val_pick:
            val_lines.append(line)
        else:
            train_lines.append(line)

    for wid, pid, x, y, w, h, text, _src in qa_rows:
        out = crops_dir / f"{wid}.png"
        _crop_word(Path(page_paths[pid]), x, y, w, h, out)
        qa_lines.append(f"{out}|{text}|{wid}")

    from scripts.data.auto_labeled_common import atomic_write_text
    atomic_write_text(normalized_dir / "train_labels.txt", "\n".join(train_lines) + ("\n" if train_lines else ""))
    atomic_write_text(normalized_dir / "val_labels.txt", "\n".join(val_lines) + ("\n" if val_lines else ""))
    atomic_write_text(normalized_dir / "qa_candidates.txt", "\n".join(qa_lines) + ("\n" if qa_lines else ""))

    return {
        "train": len(train_lines),
        "val": len(val_lines),
        "qa_candidates": len(qa_lines),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Filter labeled words and emit normalized label files.")
    p.add_argument("--db", type=Path, default=Path("data/external/auto_labeled/work.db"))
    p.add_argument("--min-conf", type=float, default=0.85)
    p.add_argument("--vocab", type=Path, default=Path("data/vocab.json"))
    p.add_argument("--existing-val", type=Path, nargs="*", default=[Path("data/combined/val_labels.txt")])
    p.add_argument("--existing-test", type=Path, nargs="*", default=[Path("data/combined/test_labels.txt")])
    p.add_argument("--existing-benchmarks", type=Path, nargs="*", default=[])
    p.add_argument(
        "--normalized-dir",
        type=Path,
        default=Path("data/external/auto_labeled/normalized"),
    )
    p.add_argument("--manifest", type=Path, default=Path("data/external/auto_labeled/manifests/manifest.json"))
    p.add_argument("--qa-per-source", type=int, default=500)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    conn = open_work_db(args.db)
    try:
        vocab = load_vocab(args.vocab)
        filter_confidence(conn, min_conf=args.min_conf)
        filter_oov(conn, vocab=vocab)
        filter_bbox(conn)
        filter_dup(conn)
        apply_text_disjoint_rule_1(
            conn,
            existing_val=args.existing_val,
            existing_test=args.existing_test,
            existing_benchmarks=args.existing_benchmarks,
        )
        promote_labeled_to_kept(conn)
        select_qa_candidates(conn, target_per_source=args.qa_per_source)
        affected = symmetric_qa_eviction(conn)
        counts = audit_disjointness(conn)
        if any(v != 0 for v in counts.values()) or affected != 0:
            print(f"AUDIT FAILED: {counts} symmetric_evicted={affected}", file=sys.stderr)
            conn.rollback()
            return 3
        emitted = emit_labels_and_crops(conn, normalized_dir=args.normalized_dir)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(
                {
                    "emitted": emitted,
                    "audit_counts": counts,
                    "min_conf": args.min_conf,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        conn.commit()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_auto_labeled_filter_emit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/data/filter_and_crop_words.py tests/unit/test_auto_labeled_filter_emit.py
git commit -m "feat(auto-labeled): filter crops, label emission, and CLI entry point"
```

---

## Task 13: Build auto-labeled stage wrapper

**Files:**
- Create: `scripts/data/build_auto_labeled_stage.py`
- Create: `tests/unit/test_auto_labeled_build_stage.py`

Thin wrapper that takes the normalized label files and produces a v8-compatible stage directory. Mirrors `scripts/data/build_v8_stage_labels.py` structure.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_auto_labeled_build_stage.py`:

```python
from __future__ import annotations

from pathlib import Path

from scripts.data.build_auto_labeled_stage import build_stage


def test_build_stage_produces_expected_files(tmp_path):
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    (normalized / "train_labels.txt").write_text("/a.png|alpha\n", encoding="utf-8")
    (normalized / "val_labels.txt").write_text("/b.png|beta\n", encoding="utf-8")

    stage_dir = tmp_path / "stages" / "stage_auto_labeled_printed"
    build_stage(
        stage_name="stage_auto_labeled_printed",
        train_labels=normalized / "train_labels.txt",
        val_labels=normalized / "val_labels.txt",
        out_dir=stage_dir,
        notes="test",
    )
    assert (stage_dir / "train_labels.txt").exists()
    assert (stage_dir / "val_labels.txt").exists()
    assert (stage_dir / "manifest.json").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_auto_labeled_build_stage.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the wrapper**

Create `scripts/data/build_auto_labeled_stage.py`:

```python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def build_stage(
    stage_name: str,
    train_labels: Path,
    val_labels: Path,
    out_dir: Path,
    notes: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(train_labels, out_dir / "train_labels.txt")
    shutil.copyfile(val_labels, out_dir / "val_labels.txt")
    manifest = {
        "stage_name": stage_name,
        "train_labels": str((out_dir / "train_labels.txt").resolve()),
        "val_labels": str((out_dir / "val_labels.txt").resolve()),
        "notes": notes,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a v8-compatible stage from auto-labeled outputs.")
    p.add_argument("--stage-name", default="stage_auto_labeled_printed")
    p.add_argument("--train-labels", type=Path, default=Path("data/external/auto_labeled/normalized/train_labels.txt"))
    p.add_argument("--val-labels", type=Path, default=Path("data/external/auto_labeled/normalized/val_labels.txt"))
    p.add_argument("--out-dir", type=Path, default=Path("data/v8/stages/stage_auto_labeled_printed"))
    p.add_argument("--notes", default="Auto-labeled Tier-1 + Tier-4 printed Hindi")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_stage(
        stage_name=args.stage_name,
        train_labels=args.train_labels,
        val_labels=args.val_labels,
        out_dir=args.out_dir,
        notes=args.notes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_auto_labeled_build_stage.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/data/build_auto_labeled_stage.py tests/unit/test_auto_labeled_build_stage.py
git commit -m "feat(auto-labeled): build_auto_labeled_stage wrapper"
```

---

## Task 14: QA tool — backend endpoints

**Files:**
- Modify: `data/labeling_tool/main.py`
- Create: `tests/unit/test_auto_labeled_qa_backend.py`

Add `GET /qa`, `POST /qa/decide`, and `GET /qa/stats` to the existing FastAPI app. Uses FastAPI's `TestClient` for tests (matches the pattern in `tests/conftest.py`).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_auto_labeled_qa_backend.py`:

```python
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from scripts.data.auto_labeled_common import open_work_db


def _make_app(db_path: Path):
    import os
    os.environ["AUTO_LABELED_WORK_DB"] = str(db_path)
    import importlib
    import data.labeling_tool.main as labeling_main
    importlib.reload(labeling_main)
    return labeling_main.app


def _seed_candidates(db_path: Path, n: int):
    conn = open_work_db(db_path)
    conn.execute(
        "INSERT INTO pages (id, source, uri, local_path, fetched_at, status)"
        " VALUES (1, 'wikipedia', 'u', 'p.png', '2026-04-16', 'labeled')"
    )
    for i in range(n):
        conn.execute(
            "INSERT INTO words (page_id, x, y, w, h, text_nfc, conf, provider, status)"
            " VALUES (1,0,0,10,10,?,0.99,'fake','exported_qa_candidate')",
            (f"cand_{i}",),
        )
    conn.commit()
    conn.close()


def test_qa_stats_initial(tmp_path):
    db = tmp_path / "work.db"
    _seed_candidates(db, 3)
    client = TestClient(_make_app(db))
    r = client.get("/qa/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 3
    assert data["decided"] == 0


def test_qa_decide_accept_advances(tmp_path):
    db = tmp_path / "work.db"
    _seed_candidates(db, 2)
    client = TestClient(_make_app(db))
    first = client.get("/qa").headers.get("x-qa-word-id")
    assert first is not None
    r = client.post("/qa/decide", data={"word_id": first, "action": "accept"})
    assert r.status_code in (200, 303)

    stats = client.get("/qa/stats").json()
    assert stats["decided"] == 1


def test_qa_decide_edit_stores_text(tmp_path):
    db = tmp_path / "work.db"
    _seed_candidates(db, 1)
    client = TestClient(_make_app(db))
    wid = client.get("/qa").headers["x-qa-word-id"]
    client.post("/qa/decide", data={"word_id": wid, "action": "edit", "edited_text": "नयाtext"})

    conn = open_work_db(db)
    row = conn.execute(
        "SELECT qa_decision, qa_edited_text FROM words WHERE id=?", (int(wid),)
    ).fetchone()
    assert row == ("edit", "नयाtext")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_auto_labeled_qa_backend.py -v`
Expected: FAIL — `/qa` endpoint missing.

- [ ] **Step 3: Extend the labeling_tool main.py**

Edit `data/labeling_tool/main.py` — replace the existing file with the following (preserves `/` and `/submit` for the manual-from-scratch flow, adds the three new endpoints):

```python
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

# --- Legacy manual-labeler (unchanged behavior) ---------------------------------

os.makedirs("templates", exist_ok=True)
LEGACY_IMAGES_DIR = "images_to_label"
os.makedirs(LEGACY_IMAGES_DIR, exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/images", StaticFiles(directory=LEGACY_IMAGES_DIR), name="images")

LEGACY_LABELS_FILE = "manual_labels.csv"


def _legacy_unlabeled() -> list[str]:
    import csv
    labeled: set[str] = set()
    if os.path.exists(LEGACY_LABELS_FILE):
        with open(LEGACY_LABELS_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    labeled.add(row[0])
    all_images = [f for f in os.listdir(LEGACY_IMAGES_DIR) if f.endswith((".png", ".jpg", ".jpeg"))]
    return [img for img in all_images if img not in labeled]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    unlabeled = _legacy_unlabeled()
    current = unlabeled[0] if unlabeled else None
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "image": current, "remaining": len(unlabeled)},
    )


@app.post("/submit", response_class=HTMLResponse)
async def submit(request: Request, filename: str = Form(...), text: str = Form(...), language: str = Form(...)):
    import csv
    if not os.path.exists(LEGACY_LABELS_FILE):
        with open(LEGACY_LABELS_FILE, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "text", "language", "font", "is_synthetic"])
    with open(LEGACY_LABELS_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([filename, text, language, "manual", False])
    return await index(request)


# --- Auto-labeled QA (new) ------------------------------------------------------

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
    # Round-robin across sources by picking the source with the fewest decided candidates so far.
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
            "qa.html",
            {"request": request, "word": None, "done": True},
        )
    resp = templates.TemplateResponse(
        "qa.html",
        {"request": request, "word": dict(row), "done": False},
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
            "SELECT COUNT(*) FROM words WHERE status='exported_qa_candidate' AND qa_decision IS NOT NULL"
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_auto_labeled_qa_backend.py -v`
Expected: All 3 PASS. (The template renders to a blank placeholder in Task 14; Task 15 adds the full template.)

- [ ] **Step 5: Commit**

```bash
git add data/labeling_tool/main.py tests/unit/test_auto_labeled_qa_backend.py
git commit -m "feat(auto-labeled): QA backend endpoints /qa, /qa/decide, /qa/stats"
```

---

## Task 15: QA tool — frontend template with keyboard bindings

**Files:**
- Create: `data/labeling_tool/templates/qa.html`

This task is pure frontend. The template must support the five keyboard shortcuts from spec §6.4 (Enter/Space=accept, e=edit, r=reject, b=back, ?=help). Tests are a single rendering smoke test via `TestClient`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_auto_labeled_qa_backend.py` (append):

```python
def test_qa_template_renders_with_keyboard_hints(tmp_path):
    db = tmp_path / "work.db"
    _seed_candidates(db, 1)
    client = TestClient(_make_app(db))
    r = client.get("/qa")
    assert r.status_code == 200
    body = r.text
    assert "Enter" in body
    assert "Accept" in body
    assert "Reject" in body
    assert "Edit" in body
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_auto_labeled_qa_backend.py::test_qa_template_renders_with_keyboard_hints -v`
Expected: FAIL — template doesn't exist yet (or renders a placeholder).

- [ ] **Step 3: Create the QA template**

Create `data/labeling_tool/templates/qa.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>QA — verify teacher predictions</title>
  <style>
    body { font-family: sans-serif; max-width: 820px; margin: 2em auto; padding: 0 1em; }
    .done { font-size: 2em; text-align: center; color: #2a7f3a; }
    .crop { background: #f4f4f4; padding: 1em; text-align: center; border-radius: 8px; }
    .crop img { max-width: 100%; image-rendering: pixelated; transform: scale(4); transform-origin: center; margin: 4em 0; }
    .pred { font-size: 2.5em; text-align: center; margin: 0.5em 0; font-weight: 600; }
    .meta { display: flex; justify-content: space-between; color: #666; margin-top: 0.5em; }
    .actions button { font-size: 1em; padding: 0.5em 1em; margin-right: 0.5em; }
    kbd { background: #eee; border-radius: 4px; padding: 0 0.3em; font-family: monospace; }
    #edit-box { display: none; margin-top: 1em; }
    #edit-box input { font-size: 1.5em; width: 100%; padding: 0.3em; }
  </style>
</head>
<body>
{% if done %}
  <div class="done">All 2000 candidates decided.<br>Click <a href="/qa/export">Export verified test labels</a>.</div>
{% else %}
  <h1>QA — {{ word.source }}</h1>
  <div class="crop">
    <img src="/images/{{ word.id }}.png" alt="word crop">
  </div>
  <div class="pred" id="pred">{{ word.text_nfc }}</div>
  <div class="meta">
    <span>Confidence: {{ "%.2f"|format(word.conf) }}</span>
    <span>Word ID: {{ word.id }}</span>
  </div>

  <form method="post" action="/qa/decide" id="decide-form">
    <input type="hidden" name="word_id" value="{{ word.id }}">
    <input type="hidden" name="action" id="action-field" value="accept">
    <div id="edit-box">
      <input type="text" name="edited_text" id="edited-text" value="{{ word.text_nfc }}">
    </div>
    <div class="actions" style="margin-top: 1em;">
      <button type="button" id="accept-btn">Accept <kbd>Enter</kbd></button>
      <button type="button" id="edit-btn">Edit <kbd>e</kbd></button>
      <button type="button" id="reject-btn">Reject <kbd>r</kbd></button>
    </div>
  </form>

  <script>
  (function () {
    const form = document.getElementById('decide-form');
    const actionField = document.getElementById('action-field');
    const editBox = document.getElementById('edit-box');
    const editInput = document.getElementById('edited-text');

    function submitWith(action) {
      actionField.value = action;
      form.submit();
    }

    document.getElementById('accept-btn').addEventListener('click', () => submitWith('accept'));
    document.getElementById('reject-btn').addEventListener('click', () => submitWith('reject'));
    document.getElementById('edit-btn').addEventListener('click', () => {
      editBox.style.display = 'block';
      editInput.focus();
    });

    document.addEventListener('keydown', (ev) => {
      const editing = editBox.style.display === 'block';
      if (editing) {
        if (ev.key === 'Enter') { submitWith('edit'); ev.preventDefault(); }
        if (ev.key === 'Escape') { editBox.style.display = 'none'; }
        return;
      }
      if (ev.key === 'Enter' || ev.key === ' ') { submitWith('accept'); ev.preventDefault(); }
      if (ev.key === 'e' || ev.key === 'E') {
        editBox.style.display = 'block';
        editInput.focus();
        ev.preventDefault();
      }
      if (ev.key === 'r' || ev.key === 'R') { submitWith('reject'); ev.preventDefault(); }
      if (ev.key === 'b' || ev.key === 'B') { window.history.back(); ev.preventDefault(); }
    });
  })();
  </script>
{% endif %}
</body>
</html>
```

- [ ] **Step 4: Run all QA-backend tests**

Run: `pytest tests/unit/test_auto_labeled_qa_backend.py -v`
Expected: All tests (including the new template test) PASS.

- [ ] **Step 5: Commit**

```bash
git add data/labeling_tool/templates/qa.html tests/unit/test_auto_labeled_qa_backend.py
git commit -m "feat(auto-labeled): QA verification template with keyboard bindings"
```

---

## Task 16: QA tool — export verified test labels

**Files:**
- Modify: `data/labeling_tool/main.py`
- Create: `tests/unit/test_auto_labeled_qa_export.py`

Adds a `GET /qa/export` endpoint that filters decisions to `accept` or `edit`, writes `verified_test_labels.txt` atomically, computes SHA256, and updates the manifest.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_auto_labeled_qa_export.py`:

```python
from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from scripts.data.auto_labeled_common import open_work_db


def _seed(conn):
    conn.execute(
        "INSERT INTO pages (id, source, uri, local_path, fetched_at, status)"
        " VALUES (1, 'wikipedia', 'u', 'p.png', '2026-04-16', 'labeled')"
    )
    rows = [
        ("a", "accept", None),
        ("b_orig", "edit", "b_edited"),
        ("c", "reject", None),
    ]
    for text, decision, edited in rows:
        conn.execute(
            "INSERT INTO words (page_id, x, y, w, h, text_nfc, conf, provider, status, qa_decision, qa_edited_text)"
            " VALUES (1,0,0,10,10,?,0.99,'fake','exported_qa_candidate',?,?)",
            (text, decision, edited),
        )


def test_export_filters_rejects_and_uses_edited_text(tmp_path):
    db = tmp_path / "work.db"
    conn = open_work_db(db)
    _seed(conn)
    conn.commit()
    conn.close()

    import os
    os.environ["AUTO_LABELED_WORK_DB"] = str(db)
    os.environ["AUTO_LABELED_NORMALIZED_DIR"] = str(tmp_path / "normalized")
    os.environ["AUTO_LABELED_MANIFEST"] = str(tmp_path / "manifest.json")
    import importlib, data.labeling_tool.main as labeling_main
    importlib.reload(labeling_main)

    client = TestClient(labeling_main.app)
    r = client.get("/qa/export")
    assert r.status_code == 200

    out = tmp_path / "normalized" / "verified_test_labels.txt"
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # reject dropped
    assert any(l.endswith("|a") for l in lines)
    assert any(l.endswith("|b_edited") for l in lines)

    manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    # Must include a sha256 string of 64 hex chars.
    import re
    assert re.search(r'"sha256":\s*"[0-9a-f]{64}"', manifest)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_auto_labeled_qa_export.py -v`
Expected: FAIL — `/qa/export` endpoint missing.

- [ ] **Step 3: Add the export endpoint**

Append to `data/labeling_tool/main.py`:

```python
# --- QA export ------------------------------------------------------------------

import hashlib
import json

DEFAULT_AUTO_NORMALIZED = Path("data/external/auto_labeled/normalized")
DEFAULT_AUTO_MANIFEST = Path("data/external/auto_labeled/manifests/manifest.json")


def _normalized_dir() -> Path:
    return Path(os.environ.get("AUTO_LABELED_NORMALIZED_DIR", str(DEFAULT_AUTO_NORMALIZED)))


def _manifest_path() -> Path:
    return Path(os.environ.get("AUTO_LABELED_MANIFEST", str(DEFAULT_AUTO_MANIFEST)))


@app.get("/qa/export")
async def qa_export():
    conn = _auto_conn()
    try:
        rows = list(
            conn.execute(
                """
                SELECT w.id, w.text_nfc, w.qa_decision, w.qa_edited_text, p.source
                FROM words w JOIN pages p ON p.id = w.page_id
                WHERE w.status='exported_qa_candidate'
                  AND w.qa_decision IN ('accept','edit')
                ORDER BY w.id
                """
            )
        )
    finally:
        conn.close()

    lines: list[str] = []
    crops_dir = _normalized_dir() / "crops"
    for wid, text, decision, edited, _source in rows:
        final_text = edited if decision == "edit" and edited else text
        img_path = crops_dir / f"{wid}.png"
        lines.append(f"{img_path}|{final_text}")

    out = _normalized_dir() / "verified_test_labels.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(lines) + ("\n" if lines else "")

    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, out)

    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    manifest_path = _manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing["verified_test_labels"] = {
        "path": str(out),
        "sha256": sha,
        "count": len(lines),
    }
    manifest_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return JSONResponse({"exported": len(lines), "sha256": sha, "path": str(out)})
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_auto_labeled_qa_export.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data/labeling_tool/main.py tests/unit/test_auto_labeled_qa_export.py
git commit -m "feat(auto-labeled): QA export verified_test_labels.txt with SHA256"
```

---

## Task 17: Orchestrator shell script + end-to-end smoke test

**Files:**
- Create: `scripts/data/run_auto_labeled_pipeline.sh`
- Create: `tests/integration/test_auto_labeled_pipeline_smoke.py`
- Create: `tests/fixtures/auto_labeled/page_wiki_001.png` (small real PNG — can be any content with visible non-white pixels)

The integration test runs the entire pipeline end-to-end against the fake provider and asserts that `train_labels.txt` contains the expected number of rows. This is the test that catches "someone renamed a CLI flag". Must run in under 5 seconds and is part of CI.

- [ ] **Step 1: Create the fixture PNG**

Run:

```bash
python -c "from PIL import Image; Image.new('L', (200, 100), 128).save('tests/fixtures/auto_labeled/page_wiki_001.png')"
```

- [ ] **Step 2: Write the failing integration test**

Create `tests/integration/test_auto_labeled_pipeline_smoke.py`:

```python
from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from scripts.data.auto_labeled_common import open_work_db


@pytest.fixture
def vocab_file(tmp_path) -> Path:
    v = tmp_path / "vocab.json"
    # Include every character present in the fake_responses fixture.
    chars = set("भारतहिंदीविकिपडयकहलो")
    v.write_text(json.dumps(sorted(chars)), encoding="utf-8")
    return v


def test_end_to_end_with_fake_provider(tmp_path, vocab_file):
    db_path = tmp_path / "work.db"
    raw_dir = tmp_path / "raw" / "wikipedia"
    raw_dir.mkdir(parents=True)

    # Seed a single fake-named page in the work DB and on disk.
    fake_page = raw_dir / "page_wiki_001.png"
    from PIL import Image
    Image.new("L", (300, 100), 200).save(fake_page)

    conn = open_work_db(db_path)
    conn.execute(
        "INSERT INTO pages (source, uri, local_path, fetched_at, status)"
        " VALUES ('wikipedia', 'u', ?, '2026-04-16', 'pending')",
        (str(fake_page),),
    )
    conn.commit()
    conn.close()

    fake_json = Path("tests/fixtures/auto_labeled/fake_responses.json").resolve()

    # Run labeler with fake provider.
    subprocess.run(
        [
            "python",
            "scripts/data/label_with_cloud_ocr.py",
            "--db", str(db_path),
            "--provider", "fake",
            "--fake-responses", str(fake_json),
            "--concurrency", "1",
        ],
        check=True,
    )

    # Run filter.
    normalized = tmp_path / "normalized"
    manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            "python",
            "scripts/data/filter_and_crop_words.py",
            "--db", str(db_path),
            "--vocab", str(vocab_file),
            "--existing-val",
            "--existing-test",
            "--normalized-dir", str(normalized),
            "--manifest", str(manifest),
            "--qa-per-source", "1",
        ],
        check=True,
    )

    train = normalized / "train_labels.txt"
    assert train.exists()
    assert len(train.read_text(encoding="utf-8").strip().splitlines()) >= 1

    m = json.loads(manifest.read_text(encoding="utf-8"))
    assert m["audit_counts"]["kept_vs_existing_val"] == 0
```

- [ ] **Step 3: Run the integration test to verify failure**

Run: `pytest tests/integration/test_auto_labeled_pipeline_smoke.py -v`
Expected: FAIL (script paths not yet wired together, or orchestrator script missing).

- [ ] **Step 4: Create the orchestrator shell script**

Create `scripts/data/run_auto_labeled_pipeline.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SOURCE=""
LIMIT=""
DB="data/external/auto_labeled/work.db"
PROVIDER="azure"
CONCURRENCY="8"
NORMALIZED_DIR="data/external/auto_labeled/normalized"
MANIFEST="data/external/auto_labeled/manifests/manifest.json"
VOCAB="data/vocab.json"

usage() {
    echo "Usage: $0 --source {wikipedia|wikisource|gov_pdf|archive_org} --limit N [--provider azure|fake] [--concurrency N]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source) SOURCE="$2"; shift 2 ;;
        --limit)  LIMIT="$2"; shift 2 ;;
        --provider) PROVIDER="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --db) DB="$2"; shift 2 ;;
        --normalized-dir) NORMALIZED_DIR="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --vocab) VOCAB="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; usage ;;
    esac
done

[[ -n "$SOURCE" && -n "$LIMIT" ]] || usage

echo "[1/4] scrape $SOURCE --limit $LIMIT"
python scripts/data/scrape_external_pages.py --source "$SOURCE" --limit "$LIMIT" --db "$DB"

echo "[2/4] label --provider $PROVIDER --concurrency $CONCURRENCY"
python scripts/data/label_with_cloud_ocr.py --db "$DB" --provider "$PROVIDER" --concurrency "$CONCURRENCY"

echo "[3/4] filter and crop"
python scripts/data/filter_and_crop_words.py \
    --db "$DB" \
    --vocab "$VOCAB" \
    --normalized-dir "$NORMALIZED_DIR" \
    --manifest "$MANIFEST"

echo "[4/4] build stage"
python scripts/data/build_auto_labeled_stage.py \
    --train-labels "$NORMALIZED_DIR/train_labels.txt" \
    --val-labels "$NORMALIZED_DIR/val_labels.txt"

echo "pipeline complete. next: start labeling_tool and verify QA candidates."
```

Then make it executable:

```bash
chmod +x scripts/data/run_auto_labeled_pipeline.sh
```

- [ ] **Step 5: Run the integration test**

Run: `pytest tests/integration/test_auto_labeled_pipeline_smoke.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/data/run_auto_labeled_pipeline.sh tests/integration/test_auto_labeled_pipeline_smoke.py tests/fixtures/auto_labeled/page_wiki_001.png
git commit -m "feat(auto-labeled): orchestrator shell + end-to-end smoke test"
```

---

## Task 18: Integration — benchmark harness and v9 curriculum

**Files:**
- Modify: `model/benchmark_eval.py`
- Modify: `scripts/training/train_v9.py`
- Create: `tests/unit/test_auto_labeled_benchmark_registration.py`

Purely additive integration: register the new suite and stage. Since this touches files whose internals we don't fully know in advance, the task includes an exploration step before modification.

- [ ] **Step 1: Explore the benchmark harness**

Run:

```bash
grep -n "SUITE\|suite\|benchmark" model/benchmark_eval.py | head -60
```

Read the suite-registration pattern. The goal is to identify the list/dict where existing suites (`synthetic_only`, `synthetic_morphed_plus_real`, `real_only`) are registered and add `auto_labeled_verified_printed` alongside them.

If the pattern is a module-level dict, the edit is a single `dict` entry. If it's a function with a hardcoded list, add a list entry. Do NOT refactor the structure — stay additive.

- [ ] **Step 2: Write a registration smoke test**

Create `tests/unit/test_auto_labeled_benchmark_registration.py`:

```python
from __future__ import annotations

import importlib


def test_auto_labeled_suite_is_registered():
    mod = importlib.import_module("model.benchmark_eval")
    # Search the module namespace for any reference to the new suite name.
    found = False
    for name in dir(mod):
        val = getattr(mod, name)
        if isinstance(val, (list, dict, tuple)):
            if "auto_labeled_verified_printed" in str(val):
                found = True
                break
    assert found, "auto_labeled_verified_printed suite not registered in model/benchmark_eval.py"
```

- [ ] **Step 3: Run the test to verify failure**

Run: `pytest tests/unit/test_auto_labeled_benchmark_registration.py -v`
Expected: FAIL.

- [ ] **Step 4: Add the suite registration**

Edit `model/benchmark_eval.py`: find the suite registry (identified in Step 1) and add the `auto_labeled_verified_printed` entry. The exact syntax depends on the existing structure, but the entry should point at `data/external/auto_labeled/normalized/verified_test_labels.txt` and include per-source breakdown reporting if the existing suites support it.

- [ ] **Step 5: Explore the v9 training curriculum**

Run:

```bash
grep -n "stage\|Stage\|curriculum" scripts/training/train_v9.py | head -60
```

Identify where stages are listed (likely a list of config dicts or paths) and insert `stage_auto_labeled_printed` between the Stage 1 synthetic entry and the Stage 2 Mozhi entry, pointing at `data/v8/stages/stage_auto_labeled_printed/`.

- [ ] **Step 6: Run the benchmark test and a curriculum smoke check**

Run: `pytest tests/unit/test_auto_labeled_benchmark_registration.py -v`
Expected: PASS.

Run:

```bash
python -c "import importlib; m = importlib.import_module('scripts.training.train_v9'); print('import_ok')"
```

Expected: `import_ok` (the stage insertion did not break module import).

- [ ] **Step 7: Commit**

```bash
git add model/benchmark_eval.py scripts/training/train_v9.py tests/unit/test_auto_labeled_benchmark_registration.py
git commit -m "feat(auto-labeled): register benchmark suite and v9 curriculum stage"
```

---

## Final verification checklist

After Task 18 is committed:

- [ ] **Run the full test suite**

```bash
pytest tests/unit/test_auto_labeled_*.py tests/integration/test_auto_labeled_pipeline_smoke.py -v
```

Expected: All tests pass.

- [ ] **Smoke the orchestrator against the fake provider**

```bash
bash scripts/data/run_auto_labeled_pipeline.sh --source wikipedia --limit 3 --provider fake
```

Expected: Pipeline completes. `data/external/auto_labeled/normalized/train_labels.txt` exists.

- [ ] **Manual real-API smoke (requires Azure credentials — not part of CI)**

```bash
export AZURE_DOC_INTEL_ENDPOINT=...
export AZURE_DOC_INTEL_KEY=...
bash scripts/data/run_auto_labeled_pipeline.sh --source wikipedia --limit 5 --provider azure
```

Expected: At least one non-empty row in `train_labels.txt`, manifest reports zero audit violations.

- [ ] **QA tool smoke**

```bash
cd data/labeling_tool && uvicorn main:app --reload &
open http://localhost:8000/qa
```

Expected: Verify one candidate with `Enter`, confirm `/qa/stats` counter advances.

- [ ] **Benchmark harness smoke**

```bash
bash scripts/run_benchmarks.sh
```

Expected: Produces an `auto_labeled_verified_printed` row (likely empty until a real pipeline run has happened, but must not crash).

Once all six §10 success criteria from the spec are green, the pipeline is shipped.

---

## Spec coverage map

Traceability from spec sections to plan tasks:

| Spec section | Task(s) |
|---|---|
| §3 Architecture overview | Task 1 (schema + layout) |
| §4.1 Scraper | Tasks 3–6 (one per source tier) |
| §4.2 Labeler | Task 7 |
| §4.3 Filter | Tasks 8–12 |
| §4.4 Build stage wrapper | Task 13 |
| §4.5 QA tool | Tasks 14–16 |
| §4.6 Orchestrator | Task 17 |
| §5 Text-disjoint policy B | Tasks 9, 10, 11 |
| §6 QA verification tool | Tasks 14, 15, 16 |
| §7 Error handling | Tasks 3–6 (fetch failure), 7 (label failure), 11 (audit failure) |
| §8 Testing strategy | Every task has a test step; Task 17 covers end-to-end |
| §9 Integration | Task 18 |
| §10 Success criteria | Final verification checklist |
| §11 Rollback | Documented in the spec; no plan task needed |
| §12 Decision log | Preserved in the spec |
| §13 Open items | Gov-seed URL list (Task 5), provider credentials (prerequisite block), Wikipedia article-selection prototyping (manual verification step) |

No spec section is unimplemented by this plan.
