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
    _seed(
        conn,
        [
            (1, "uniq_w_1", "kept"),
            (1, "uniq_w_2", "kept"),
            (1, "uniq_w_3", "kept"),
            (1, "dup_text", "kept"),
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
