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
        else:
            print(f"source {args.source} not yet implemented", file=sys.stderr)
            return 2
        conn.commit()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
