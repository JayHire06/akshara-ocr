from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from scripts.data.auto_labeled_common import normalize_text, open_work_db
from scripts.data.auto_labeled_provider import Provider, WordBox, get_provider

# Pages per SQLite commit. Small enough that a crash loses at most a few
# minutes of Azure work; large enough that commit overhead stays <1% of
# per-page cost.
COMMIT_BATCH_SIZE = 500


async def _label_one(
    provider: Provider,
    page_id: int,
    local_path: Path,
    max_attempts: int,
) -> tuple[int, list[WordBox] | None, str | None]:
    last_exc: str | None = None
    for attempt in range(max_attempts):
        try:
            boxes = await asyncio.to_thread(provider.label_page, local_path)
            return page_id, boxes, None
        except Exception as exc:
            last_exc = str(exc)[:500]
            if attempt + 1 < max_attempts:
                await asyncio.sleep(min(2 ** attempt, 30))
    return page_id, None, last_exc


async def _drain(
    pending: list[tuple[int, str]],
    provider: Provider,
    concurrency: int,
    max_attempts: int,
) -> list[tuple[int, list[WordBox] | None, str | None]]:
    sem = asyncio.Semaphore(concurrency)

    async def _guarded(page_id: int, local_path: str):
        async with sem:
            return await _label_one(provider, page_id, Path(local_path), max_attempts)

    return await asyncio.gather(*(_guarded(pid, p) for pid, p in pending))


def run_labeler(
    db_path: Path,
    provider: Provider,
    concurrency: int = 8,
    max_attempts: int = 3,
) -> None:
    conn = open_work_db(db_path)
    try:
        pending = list(
            conn.execute("SELECT id, local_path FROM pages WHERE status = 'pending'")
        )
        if not pending:
            return
        results = asyncio.run(_drain(pending, provider, concurrency, max_attempts))

        pages_in_batch = 0
        for page_id, boxes, err in results:
            if err is not None or boxes is None:
                conn.execute(
                    "UPDATE pages SET status='label_failed', error=? WHERE id=?",
                    (err, page_id),
                )
            else:
                for b in boxes:
                    text = normalize_text(b.text)
                    if not text:
                        continue
                    conn.execute(
                        "INSERT INTO words (page_id, x, y, w, h, text_nfc, conf, provider, status)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'labeled')",
                        (page_id, b.x, b.y, b.w, b.h, text, b.conf, provider.name),
                    )
                conn.execute("UPDATE pages SET status='labeled' WHERE id=?", (page_id,))
            pages_in_batch += 1
            if pages_in_batch >= COMMIT_BATCH_SIZE:
                conn.commit()
                pages_in_batch = 0
        conn.commit()
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Label pending pages via a cloud OCR provider.")
    p.add_argument("--db", type=Path, default=Path("data/external/auto_labeled/work.db"))
    p.add_argument("--provider", choices=("azure", "fake"), default="azure")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--fake-responses", type=Path, help="JSON fixture for --provider fake")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    provider = get_provider(
        args.provider,
        fake_responses_path=args.fake_responses,
    )
    run_labeler(db_path=args.db, provider=provider, concurrency=args.concurrency)
    return 0


if __name__ == "__main__":
    sys.exit(main())
