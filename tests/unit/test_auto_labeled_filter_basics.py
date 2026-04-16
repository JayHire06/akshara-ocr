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
    _seed_word(conn, 1, "A", 0.99, x=10, y=10, w=50, h=20)
    _seed_word(conn, 1, "B", 0.99, x=990, y=990, w=50, h=50)
    conn.commit()
    filter_bbox(conn, page_dims={1: (100, 100)})
    rows = dict(conn.execute("SELECT text_nfc, status FROM words"))
    assert rows["A"] == "labeled"
    assert rows["B"] == "filtered_invalid_bbox"


def test_filter_dup(tmp_path):
    conn = open_work_db(tmp_path / "w.db")
    _seed_page(conn, 1)
    _seed_word(conn, 1, "भारत", 0.99, x=10, y=10)
    _seed_word(conn, 1, "भारत", 0.99, x=10, y=10)
    _seed_word(conn, 1, "भारत", 0.99, x=50, y=10)
    conn.commit()
    filter_dup(conn)
    statuses = sorted(row[0] for row in conn.execute("SELECT status FROM words ORDER BY id"))
    assert statuses.count("labeled") == 2
    assert statuses.count("filtered_dup") == 1
