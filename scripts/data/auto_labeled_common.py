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
