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
