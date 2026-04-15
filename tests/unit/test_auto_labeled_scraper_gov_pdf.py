from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.data.auto_labeled_common import open_work_db
from scripts.data.scrape_external_pages import scrape_gov_pdf


def test_scrape_gov_pdf_rasterizes_and_inserts(tmp_path, monkeypatch):
    db_path = tmp_path / "work.db"
    raw_dir = tmp_path / "raw" / "gov_pdf"
    raw_dir.mkdir(parents=True)

    # Provide an in-memory seed list (one fake URL).
    seeds = tmp_path / "seeds.txt"
    seeds.write_text("https://example.gov.in/test.pdf\n", encoding="utf-8")

    # Stub downloader to write fake PDF bytes.
    def fake_download(url, dest):
        Path(dest).write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._download_pdf", fake_download
    )

    # Stub rasterizer to yield two PIL images.
    def fake_rasterize(pdf_path):
        return [Image.new("RGB", (800, 1000), "white"), Image.new("RGB", (800, 1000), "white")]

    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._rasterize_pdf", fake_rasterize
    )
    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._is_non_blank_png", lambda path: True
    )

    conn = open_work_db(db_path)
    scrape_gov_pdf(conn=conn, limit=2, raw_dir=raw_dir, seeds_path=seeds)
    conn.commit()

    rows = list(conn.execute("SELECT source, status FROM pages"))
    assert len(rows) == 2
    assert all(r == ("gov_pdf", "pending") for r in rows)

    # PDF file should have been cleaned up after rasterization; only PNGs remain.
    assert not list(raw_dir.glob("*.pdf"))
    assert len(list(raw_dir.glob("*.png"))) == 2


def test_scrape_gov_pdf_marks_blank_pages_as_fetch_failed_and_unlinks(tmp_path, monkeypatch):
    db_path = tmp_path / "work.db"
    raw_dir = tmp_path / "raw" / "gov_pdf"
    raw_dir.mkdir(parents=True)

    seeds = tmp_path / "seeds.txt"
    seeds.write_text("https://example.gov.in/blank.pdf\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._download_pdf",
        lambda url, dest: Path(dest).write_bytes(b"%PDF-1.4 fake"),
    )
    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._rasterize_pdf",
        lambda pdf_path: [Image.new("RGB", (800, 1000), "white")],
    )
    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._is_non_blank_png",
        lambda path: False,
    )

    conn = open_work_db(db_path)
    scrape_gov_pdf(conn=conn, limit=1, raw_dir=raw_dir, seeds_path=seeds)
    conn.commit()

    row = conn.execute("SELECT status, error FROM pages").fetchone()
    assert row == ("fetch_failed", "blank_render")
    # Orphaned PNG and the .pdf must both be cleaned up.
    assert not list(raw_dir.glob("*.png"))
    assert not list(raw_dir.glob("*.pdf"))


def test_scrape_gov_pdf_records_zero_pages_when_rasterizer_returns_empty(tmp_path, monkeypatch):
    db_path = tmp_path / "work.db"
    raw_dir = tmp_path / "raw" / "gov_pdf"
    raw_dir.mkdir(parents=True)

    seeds = tmp_path / "seeds.txt"
    seeds.write_text("https://example.gov.in/empty.pdf\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._download_pdf",
        lambda url, dest: Path(dest).write_bytes(b"%PDF-1.4 fake"),
    )
    monkeypatch.setattr(
        "scripts.data.scrape_external_pages._rasterize_pdf",
        lambda pdf_path: [],
    )

    conn = open_work_db(db_path)
    scrape_gov_pdf(conn=conn, limit=1, raw_dir=raw_dir, seeds_path=seeds)
    conn.commit()

    row = conn.execute("SELECT status, error FROM pages").fetchone()
    assert row == ("fetch_failed", "zero_pages")
    assert not list(raw_dir.glob("*.pdf"))
