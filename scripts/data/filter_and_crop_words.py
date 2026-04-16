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
    failed_pages: set[int] = set()
    if page_dims is None:
        page_dims = {}
        for pid, lp in conn.execute("SELECT id, local_path FROM pages"):
            try:
                with Image.open(lp) as im:
                    page_dims[pid] = im.size
            except Exception:
                page_dims[pid] = (0, 0)
                failed_pages.add(pid)

    # Separate "real OOB bbox" from "page file failed to open" so downstream
    # audits can distinguish a genuine bad label from a missing/corrupt page.
    bbox_bad: list[int] = []
    page_bad: list[int] = []
    for wid, pid, x, y, w, h in conn.execute(
        "SELECT id, page_id, x, y, w, h FROM words WHERE status='labeled'"
    ):
        if pid in failed_pages:
            page_bad.append(wid)
            continue
        W, H = page_dims.get(pid, (0, 0))
        x2 = min(x + w, W)
        y2 = min(y + h, H)
        x = max(x, 0)
        y = max(y, 0)
        if x2 - x <= 0 or y2 - y <= 0 or W == 0 or H == 0:
            bbox_bad.append(wid)
    if bbox_bad:
        conn.executemany(
            "UPDATE words SET status='filtered_invalid_bbox', filter_reason='bbox_oob' WHERE id=?",
            [(i,) for i in bbox_bad],
        )
    if page_bad:
        conn.executemany(
            "UPDATE words SET status='filtered_invalid_bbox', filter_reason='page_open_failed' WHERE id=?",
            [(i,) for i in page_bad],
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
