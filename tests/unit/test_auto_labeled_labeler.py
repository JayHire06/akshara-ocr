from __future__ import annotations

from pathlib import Path

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
