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
            "INSERT INTO words"
            " (page_id, x, y, w, h, text_nfc, conf, provider, status, qa_decision, qa_edited_text)"
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
    import importlib
    import data.labeling_tool.main as labeling_main
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
    import re
    assert re.search(r'"sha256":\s*"[0-9a-f]{64}"', manifest)
