from __future__ import annotations

from PIL import Image

from scripts.data.auto_labeled_common import open_work_db
from scripts.data.scrape_external_pages import scrape_archive_org


def test_scrape_archive_org_inserts_pending(tmp_path, monkeypatch):
    db_path = tmp_path / "work.db"
    raw_dir = tmp_path / "raw" / "archive_org"
    raw_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._search_archive_hindi_text_items",
        lambda limit_items: [("item-001", "https://archive.org/download/item-001/item-001_jp2.zip")],
    )

    def fake_fetch(url, max_pages):
        return [Image.new("L", (900, 1200), 255) for _ in range(max_pages)]

    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._fetch_and_extract_jp2_pages", fake_fetch
    )
    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._is_non_blank_png", lambda path: True
    )

    conn = open_work_db(db_path)
    scrape_archive_org(conn=conn, limit=3, raw_dir=raw_dir)
    conn.commit()

    rows = list(conn.execute("SELECT source, status FROM pages"))
    assert len(rows) == 3
    assert all(r == ("archive_org", "pending") for r in rows)
