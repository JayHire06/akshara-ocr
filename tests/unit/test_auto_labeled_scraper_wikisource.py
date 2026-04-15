from __future__ import annotations

from scripts.data.auto_labeled_common import open_work_db
from scripts.data.scrape_external_pages import scrape_wikisource


def test_scrape_wikisource_inserts_pending_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "work.db"
    raw_dir = tmp_path / "raw" / "wikisource"
    raw_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._random_wikisource_page_urls",
        lambda limit: ["https://hi.wikisource.org/wiki/पृष्ठ:A/1"][:limit],
    )
    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._screenshot_url",
        lambda url, out_path: out_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100),
    )
    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._is_non_blank_png",
        lambda path: True,
    )

    conn = open_work_db(db_path)
    scrape_wikisource(conn=conn, limit=1, raw_dir=raw_dir)
    conn.commit()

    row = conn.execute("SELECT source, status, local_path FROM pages").fetchone()
    assert row[0] == "wikisource"
    assert row[1] == "pending"
    assert row[2] and str(raw_dir) in row[2]
