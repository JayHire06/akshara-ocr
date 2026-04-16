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
    val_texts = ["भारत "]
    pairs = levenshtein_audit_sample(kept_texts, val_texts, sample_size=10)
    assert len(pairs) >= 1
    assert any(p[0] == "भारत" for p in pairs)
