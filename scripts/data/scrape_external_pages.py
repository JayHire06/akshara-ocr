from __future__ import annotations

import argparse
import datetime as _dt
import sqlite3
import sys
import uuid
from pathlib import Path

import httpx
from PIL import Image

from scripts.data.auto_labeled_common import open_work_db

WIKIPEDIA_API = "https://hi.wikipedia.org/w/api.php"
WIKIPEDIA_BASE = "https://hi.wikipedia.org/wiki/"

SOURCES = ("wikipedia", "wikisource", "gov_pdf", "archive_org")


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _is_non_blank_png(path: Path) -> bool:
    """Return True if the PNG is openable and has at least 5% non-white pixels."""
    try:
        with Image.open(path) as im:
            im = im.convert("L")
            pixels = list(im.getdata())
    except Exception:
        return False
    if not pixels:
        return False
    non_white = sum(1 for p in pixels if p < 240)
    return (non_white / len(pixels)) >= 0.05


def _insert_page(
    conn: sqlite3.Connection,
    source: str,
    uri: str,
    local_path: Path,
    status: str,
    error: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO pages (source, uri, local_path, fetched_at, status, error)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (source, uri, str(local_path), _now_iso(), status, error),
    )


# --- Wikipedia ------------------------------------------------------------------


def _random_wikipedia_article_urls(limit: int) -> list[str]:
    """Use the MediaWiki API to pick `limit` random article URLs from hi.wikipedia."""
    with httpx.Client(timeout=30.0) as client:
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": min(limit, 10),
        }
        urls: list[str] = []
        empty_batches = 0
        while len(urls) < limit:
            resp = client.get(WIKIPEDIA_API, params=params)
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("query", {}).get("random", [])
            if not batch:
                empty_batches += 1
                if empty_batches >= 3:
                    break
                continue
            empty_batches = 0
            for item in batch:
                title = item["title"].replace(" ", "_")
                urls.append(WIKIPEDIA_BASE + title)
                if len(urls) >= limit:
                    break
    return urls[:limit]


def _screenshot_url(url: str, out_path: Path) -> None:
    """Render a URL with Playwright and save a full-page screenshot."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 2000})
            page.goto(url, wait_until="networkidle", timeout=60_000)
            page.screenshot(path=str(out_path), full_page=True)
        finally:
            browser.close()


def scrape_wikipedia(conn: sqlite3.Connection, limit: int, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    urls = _random_wikipedia_article_urls(limit)
    for url in urls:
        out = raw_dir / f"{uuid.uuid4().hex}.png"
        try:
            _screenshot_url(url, out)
            if not _is_non_blank_png(out):
                out.unlink(missing_ok=True)
                _insert_page(conn, "wikipedia", url, out, "fetch_failed", error="blank_render")
                continue
            _insert_page(conn, "wikipedia", url, out, "pending")
        except Exception as exc:
            _insert_page(conn, "wikipedia", url, out, "fetch_failed", error=str(exc)[:500])


# --- Wikisource -----------------------------------------------------------------

WIKISOURCE_API = "https://hi.wikisource.org/w/api.php"
WIKISOURCE_BASE = "https://hi.wikisource.org/wiki/"


def _random_wikisource_page_urls(limit: int) -> list[str]:
    """Use the MediaWiki API on hi.wikisource to sample random pages in the Page: (ProofreadPage) namespace."""
    with httpx.Client(timeout=30.0) as client:
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnnamespace": 104,  # Page namespace for ProofreadPage
            "rnlimit": min(limit, 10),
        }
        urls: list[str] = []
        empty_batches = 0
        while len(urls) < limit:
            resp = client.get(WIKISOURCE_API, params=params)
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("query", {}).get("random", [])
            if not batch:
                empty_batches += 1
                if empty_batches >= 3:
                    break
                continue
            empty_batches = 0
            for item in batch:
                title = item["title"].replace(" ", "_")
                urls.append(WIKISOURCE_BASE + title)
                if len(urls) >= limit:
                    break
    return urls[:limit]


def scrape_wikisource(conn: sqlite3.Connection, limit: int, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    urls = _random_wikisource_page_urls(limit)
    for url in urls:
        out = raw_dir / f"{uuid.uuid4().hex}.png"
        try:
            _screenshot_url(url, out)
            if not _is_non_blank_png(out):
                out.unlink(missing_ok=True)
                _insert_page(conn, "wikisource", url, out, "fetch_failed", error="blank_render")
                continue
            _insert_page(conn, "wikisource", url, out, "pending")
        except Exception as exc:
            _insert_page(conn, "wikisource", url, out, "fetch_failed", error=str(exc)[:500])


# --- Gov PDFs -------------------------------------------------------------------

DEFAULT_GOV_SEEDS = Path(__file__).parent / "auto_labeled_gov_seeds.txt"


def _load_seeds(path: Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _download_pdf(url: str, dest: Path) -> None:
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        Path(dest).write_bytes(resp.content)


def _rasterize_pdf(pdf_path: Path) -> list:
    from pdf2image import convert_from_path
    return convert_from_path(str(pdf_path), dpi=300)


def scrape_gov_pdf(
    conn: sqlite3.Connection,
    limit: int,
    raw_dir: Path,
    seeds_path: Path = DEFAULT_GOV_SEEDS,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    seeds = _load_seeds(seeds_path)
    if not seeds:
        return
    pages_written = 0
    for url in seeds:
        if pages_written >= limit:
            break
        pdf_local = raw_dir / f"{uuid.uuid4().hex}.pdf"
        try:
            try:
                _download_pdf(url, pdf_local)
                images = _rasterize_pdf(pdf_local)
            except Exception as exc:
                _insert_page(conn, "gov_pdf", url, pdf_local, "fetch_failed", error=str(exc)[:500])
                continue
            if not images:
                _insert_page(conn, "gov_pdf", url, pdf_local, "fetch_failed", error="zero_pages")
                continue
            for img in images:
                if pages_written >= limit:
                    break
                out = raw_dir / f"{uuid.uuid4().hex}.png"
                img.save(out, format="PNG")
                if not _is_non_blank_png(out):
                    out.unlink(missing_ok=True)
                    _insert_page(conn, "gov_pdf", url, out, "fetch_failed", error="blank_render")
                    continue
                _insert_page(conn, "gov_pdf", url, out, "pending")
                pages_written += 1
        finally:
            pdf_local.unlink(missing_ok=True)


# --- archive.org ---------------------------------------------------------------

ARCHIVE_SEARCH_URL = "https://archive.org/advancedsearch.php"
ARCHIVE_MAX_ITEM_BYTES = 300 * 1024 * 1024  # 300 MB cap per item


def _search_archive_hindi_text_items(limit_items: int) -> list[tuple[str, str]]:
    """Return list of (identifier, jp2_zip_url) for Hindi text items."""
    params = {
        "q": "language:hin AND mediatype:texts",
        "fl[]": ["identifier"],
        "rows": limit_items,
        "page": 1,
        "output": "json",
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(ARCHIVE_SEARCH_URL, params=params)
        resp.raise_for_status()
        docs = resp.json().get("response", {}).get("docs", [])
    out: list[tuple[str, str]] = []
    for d in docs:
        ident = d["identifier"]
        jp2 = f"https://archive.org/download/{ident}/{ident}_jp2.zip"
        out.append((ident, jp2))
    return out


def _fetch_and_extract_jp2_pages(url: str, max_pages: int) -> list:
    """Download a _jp2.zip and return up to max_pages PIL images."""
    import io
    import zipfile
    from PIL import Image as _PILImage

    with httpx.Client(timeout=600.0, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            size = int(resp.headers.get("content-length", 0))
            if size and size > ARCHIVE_MAX_ITEM_BYTES:
                raise RuntimeError(f"item too large: {size} bytes")
            buf = io.BytesIO()
            for chunk in resp.iter_bytes():
                buf.write(chunk)
                if buf.tell() > ARCHIVE_MAX_ITEM_BYTES:
                    raise RuntimeError("item exceeded cap during stream")
    buf.seek(0)
    images: list = []
    with zipfile.ZipFile(buf) as zf:
        jp2_names = sorted(n for n in zf.namelist() if n.endswith(".jp2"))[:max_pages]
        for name in jp2_names:
            with zf.open(name) as jf:
                images.append(_PILImage.open(io.BytesIO(jf.read())).convert("L"))
    return images


def scrape_archive_org(conn: sqlite3.Connection, limit: int, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    items_needed = max(1, limit // 20)
    items = _search_archive_hindi_text_items(items_needed + 5)
    pages_written = 0
    for ident, url in items:
        if pages_written >= limit:
            break
        try:
            imgs = _fetch_and_extract_jp2_pages(url, max_pages=min(20, limit - pages_written))
        except Exception as exc:
            _insert_page(conn, "archive_org", url, raw_dir / f"{ident}.skip", "fetch_failed", error=str(exc)[:500])
            continue
        if not imgs:
            _insert_page(conn, "archive_org", url, raw_dir / f"{ident}.skip", "fetch_failed", error="zero_pages")
            continue
        for img in imgs:
            if pages_written >= limit:
                break
            out = raw_dir / f"{uuid.uuid4().hex}.png"
            img.save(out, format="PNG")
            if not _is_non_blank_png(out):
                out.unlink(missing_ok=True)
                _insert_page(conn, "archive_org", url, out, "fetch_failed", error="blank_render")
                continue
            _insert_page(conn, "archive_org", url, out, "pending")
            pages_written += 1


# --- CLI entry point -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scrape external printed Hindi pages into the work queue.")
    p.add_argument("--source", required=True, choices=SOURCES)
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--db", type=Path, default=Path("data/external/auto_labeled/work.db"))
    p.add_argument("--raw-root", type=Path, default=Path("data/external/auto_labeled/raw"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    conn = open_work_db(args.db)
    try:
        raw_dir = args.raw_root / args.source
        if args.source == "wikipedia":
            scrape_wikipedia(conn=conn, limit=args.limit, raw_dir=raw_dir)
        elif args.source == "wikisource":
            scrape_wikisource(conn=conn, limit=args.limit, raw_dir=raw_dir)
        elif args.source == "gov_pdf":
            scrape_gov_pdf(conn=conn, limit=args.limit, raw_dir=raw_dir)
        elif args.source == "archive_org":
            scrape_archive_org(conn=conn, limit=args.limit, raw_dir=raw_dir)
        else:
            print(f"source {args.source} not yet implemented", file=sys.stderr)
            return 2
        conn.commit()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
