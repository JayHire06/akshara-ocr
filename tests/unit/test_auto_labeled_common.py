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
