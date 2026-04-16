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
    r = client.post("/qa/decide", data={"word_id": first, "action": "accept"}, follow_redirects=False)
    assert r.status_code in (200, 303)

    stats = client.get("/qa/stats").json()
    assert stats["decided"] == 1


def test_qa_decide_edit_stores_text(tmp_path):
    db = tmp_path / "work.db"
    _seed_candidates(db, 1)
    client = TestClient(_make_app(db))
    wid = client.get("/qa").headers["x-qa-word-id"]
    client.post(
        "/qa/decide",
        data={"word_id": wid, "action": "edit", "edited_text": "नयाtext"},
        follow_redirects=False,
    )

    conn = open_work_db(db)
    row = conn.execute(
        "SELECT qa_decision, qa_edited_text FROM words WHERE id=?", (int(wid),)
    ).fetchone()
    assert row == ("edit", "नयाtext")
