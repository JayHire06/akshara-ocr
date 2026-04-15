from __future__ import annotations

from pathlib import Path

from scripts.data.auto_labeled_common import open_work_db
from scripts.data.scrape_external_pages import scrape_wikipedia


def test_scrape_wikipedia_inserts_pending_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "work.db"
    raw_dir = tmp_path / "raw" / "wikipedia"
    raw_dir.mkdir(parents=True)

    fake_urls = ["https://hi.wikipedia.org/wiki/एक", "https://hi.wikipedia.org/wiki/दो"]

    def fake_random_article_urls(limit):
        return fake_urls[:limit]

    def fake_screenshot(url, out_path):
        out_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._random_wikipedia_article_urls",
        fake_random_article_urls,
    )
    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._screenshot_url",
        fake_screenshot,
    )
    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._is_non_blank_png",
        lambda path: True,
    )

    conn = open_work_db(db_path)
    scrape_wikipedia(conn=conn, limit=2, raw_dir=raw_dir)
    conn.commit()

    rows = list(conn.execute("SELECT source, uri, status FROM pages ORDER BY id"))
    assert len(rows) == 2
    assert all(r[0] == "wikipedia" for r in rows)
    assert rows[0][1] == fake_urls[0]
    assert all(r[2] == "pending" for r in rows)

    # Files were actually written.
    paths = [Path(r[0]) for r in conn.execute("SELECT local_path FROM pages")]
    assert all(p.exists() for p in paths)


def test_scrape_wikipedia_marks_blank_renders_as_fetch_failed(tmp_path, monkeypatch):
    db_path = tmp_path / "work.db"
    raw_dir = tmp_path / "raw" / "wikipedia"
    raw_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._random_wikipedia_article_urls",
        lambda limit: ["https://hi.wikipedia.org/wiki/Blank"],
    )
    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._screenshot_url",
        lambda url, out_path: out_path.write_bytes(b""),
    )
    # Force the non-blank check to report blank.
    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._is_non_blank_png",
        lambda path: False,
    )

    conn = open_work_db(db_path)
    scrape_wikipedia(conn=conn, limit=1, raw_dir=raw_dir)
    conn.commit()

    status = conn.execute("SELECT status FROM pages").fetchone()[0]
    assert status == "fetch_failed"
