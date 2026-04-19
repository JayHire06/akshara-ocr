from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from PIL import Image

from scripts.data.auto_labeled_common import load_text_strings, open_work_db


def filter_confidence(conn: sqlite3.Connection, min_conf: float) -> None:
    conn.execute(
        "UPDATE words SET status='filtered_low_conf', filter_reason=?"
        " WHERE status='labeled' AND conf < ?",
        (f"conf<{min_conf}", min_conf),
    )


def filter_oov(conn: sqlite3.Connection, vocab: set[str]) -> None:
    rows = list(conn.execute("SELECT id, text_nfc FROM words WHERE status='labeled'"))
    bad_ids: list[int] = []
    for wid, text in rows:
        if any(ch not in vocab for ch in text):
            bad_ids.append(wid)
    if bad_ids:
        conn.executemany(
            "UPDATE words SET status='filtered_oov', filter_reason='oov_char' WHERE id=?",
            [(i,) for i in bad_ids],
        )


def filter_bbox(
    conn: sqlite3.Connection,
    page_dims: dict[int, tuple[int, int]] | None = None,
) -> None:
    failed_pages: set[int] = set()
    if page_dims is None:
        page_dims = {}
        for pid, lp in conn.execute("SELECT id, local_path FROM pages"):
            try:
                with Image.open(lp) as im:
                    page_dims[pid] = im.size
            except Exception:
                page_dims[pid] = (0, 0)
                failed_pages.add(pid)

    # Separate "real OOB bbox" from "page file failed to open" so downstream
    # audits can distinguish a genuine bad label from a missing/corrupt page.
    bbox_bad: list[int] = []
    page_bad: list[int] = []
    for wid, pid, x, y, w, h in conn.execute(
        "SELECT id, page_id, x, y, w, h FROM words WHERE status='labeled'"
    ):
        if pid in failed_pages:
            page_bad.append(wid)
            continue
        W, H = page_dims.get(pid, (0, 0))
        x2 = min(x + w, W)
        y2 = min(y + h, H)
        x = max(x, 0)
        y = max(y, 0)
        if x2 - x <= 0 or y2 - y <= 0 or W == 0 or H == 0:
            bbox_bad.append(wid)
    if bbox_bad:
        conn.executemany(
            "UPDATE words SET status='filtered_invalid_bbox', filter_reason='bbox_oob' WHERE id=?",
            [(i,) for i in bbox_bad],
        )
    if page_bad:
        conn.executemany(
            "UPDATE words SET status='filtered_invalid_bbox', filter_reason='page_open_failed' WHERE id=?",
            [(i,) for i in page_bad],
        )


def filter_dup(conn: sqlite3.Connection) -> None:
    """Keep the earliest (lowest id) row for each (page_id, text_nfc, x, y) tuple; mark the rest as dup."""
    dup_ids: list[int] = []
    seen: set[tuple[int, str, int, int]] = set()
    for wid, pid, text, x, y in conn.execute(
        "SELECT id, page_id, text_nfc, x, y FROM words WHERE status='labeled' ORDER BY id"
    ):
        key = (pid, text, x, y)
        if key in seen:
            dup_ids.append(wid)
        else:
            seen.add(key)
    if dup_ids:
        conn.executemany(
            "UPDATE words SET status='filtered_dup', filter_reason='dup' WHERE id=?",
            [(i,) for i in dup_ids],
        )


def _load_all_strings(
    val: list[Path],
    test: list[Path],
    bench: list[Path],
) -> tuple[set[str], set[str], set[str]]:
    return (
        load_text_strings(val),
        load_text_strings(test),
        load_text_strings(bench),
    )


def apply_text_disjoint_rule_1(
    conn: sqlite3.Connection,
    existing_val: list[Path],
    existing_test: list[Path],
    existing_benchmarks: list[Path],
) -> None:
    val, test, bench = _load_all_strings(existing_val, existing_test, existing_benchmarks)

    conn.execute("DROP TABLE IF EXISTS temp.existing_val_strings")
    conn.execute("DROP TABLE IF EXISTS temp.existing_test_strings")
    conn.execute("DROP TABLE IF EXISTS temp.existing_bench_strings")
    conn.execute("CREATE TEMP TABLE existing_val_strings (text_nfc TEXT PRIMARY KEY)")
    conn.execute("CREATE TEMP TABLE existing_test_strings (text_nfc TEXT PRIMARY KEY)")
    conn.execute("CREATE TEMP TABLE existing_bench_strings (text_nfc TEXT PRIMARY KEY)")
    conn.executemany("INSERT OR IGNORE INTO existing_val_strings VALUES (?)", [(s,) for s in val])
    conn.executemany("INSERT OR IGNORE INTO existing_test_strings VALUES (?)", [(s,) for s in test])
    conn.executemany("INSERT OR IGNORE INTO existing_bench_strings VALUES (?)", [(s,) for s in bench])

    conn.execute(
        """
        UPDATE words
        SET status='filtered_text_leak',
            filter_reason='leak_vs_existing_val_test_bench'
        WHERE status='labeled'
          AND text_nfc IN (
              SELECT text_nfc FROM existing_val_strings
              UNION
              SELECT text_nfc FROM existing_test_strings
              UNION
              SELECT text_nfc FROM existing_bench_strings
          )
        """
    )


def promote_labeled_to_kept(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE words SET status='kept' WHERE status='labeled'")


def select_qa_candidates(conn: sqlite3.Connection, target_per_source: int = 500) -> None:
    """Promote kept rows to exported_qa_candidate using unique-text predicate.

    Assumes existing_val_strings, existing_test_strings, existing_bench_strings temp
    tables exist from apply_text_disjoint_rule_1; if not, creates empty ones so the
    query still works.
    """
    for name in ("existing_val_strings", "existing_test_strings", "existing_bench_strings"):
        conn.execute(f"CREATE TEMP TABLE IF NOT EXISTS {name} (text_nfc TEXT PRIMARY KEY)")

    conn.execute(
        """
        WITH unique_text_kept AS (
            SELECT id, text_nfc, page_id
            FROM words
            WHERE status = 'kept'
              AND text_nfc IN (
                  SELECT text_nfc FROM words WHERE status = 'kept'
                  GROUP BY text_nfc HAVING COUNT(*) = 1
              )
              AND text_nfc NOT IN (SELECT text_nfc FROM existing_val_strings)
              AND text_nfc NOT IN (SELECT text_nfc FROM existing_test_strings)
              AND text_nfc NOT IN (SELECT text_nfc FROM existing_bench_strings)
        ),
        ranked AS (
            SELECT u.id, u.text_nfc, p.source,
                   ROW_NUMBER() OVER (PARTITION BY p.source ORDER BY RANDOM()) AS rn
            FROM unique_text_kept u JOIN pages p ON p.id = u.page_id
        )
        UPDATE words
        SET status='exported_qa_candidate'
        WHERE id IN (SELECT id FROM ranked WHERE rn <= ?)
        """,
        (target_per_source,),
    )


def symmetric_qa_eviction(conn: sqlite3.Connection) -> int:
    """Belt-and-suspenders: evict any kept row whose text matches an exported candidate.

    Returns the number of rows affected. In a correct run this must be zero.
    """
    cur = conn.execute(
        """
        UPDATE words
        SET status='filtered_text_leak',
            filter_reason='leak_vs_qa_candidate'
        WHERE status='kept'
          AND text_nfc IN (
              SELECT DISTINCT text_nfc FROM words WHERE status='exported_qa_candidate'
          )
        """
    )
    return cur.rowcount


def audit_disjointness(conn: sqlite3.Connection) -> dict[str, int]:
    """Return the four audit counts. All must be zero in a correct run."""
    def _count(sql: str) -> int:
        return conn.execute(sql).fetchone()[0]

    return {
        "kept_vs_existing_val": _count(
            "SELECT COUNT(*) FROM words"
            " WHERE status='kept' AND text_nfc IN (SELECT text_nfc FROM existing_val_strings)"
        ),
        "kept_vs_existing_test": _count(
            "SELECT COUNT(*) FROM words"
            " WHERE status='kept' AND text_nfc IN (SELECT text_nfc FROM existing_test_strings)"
        ),
        "kept_vs_existing_bench": _count(
            "SELECT COUNT(*) FROM words"
            " WHERE status='kept' AND text_nfc IN (SELECT text_nfc FROM existing_bench_strings)"
        ),
        "kept_vs_qa_candidate": _count(
            "SELECT COUNT(*) FROM words w1"
            " WHERE w1.status='kept'"
            " AND w1.text_nfc IN (SELECT text_nfc FROM words WHERE status='exported_qa_candidate')"
        ),
    }


def levenshtein_audit_sample(
    kept_texts: list[str],
    val_texts: list[str],
    sample_size: int = 100,
) -> list[tuple[str, str, int]]:
    """Return up to sample_size (kept, val, distance) triples where Levenshtein <= 1."""
    import random
    try:
        from Levenshtein import distance as lev  # type: ignore
    except ImportError:
        # Pure-Python fallback. O(n*m) per call, fine for the bounded
        # (<= 500 x <= 500) samples this audit draws.
        def lev(a: str, b: str) -> int:
            if a == b:
                return 0
            if not a:
                return len(b)
            if not b:
                return len(a)
            prev = list(range(len(b) + 1))
            for i, ca in enumerate(a, start=1):
                curr = [i] + [0] * len(b)
                for j, cb in enumerate(b, start=1):
                    curr[j] = min(
                        curr[j - 1] + 1,
                        prev[j] + 1,
                        prev[j - 1] + (0 if ca == cb else 1),
                    )
                prev = curr
            return prev[-1]

    out: list[tuple[str, str, int]] = []
    kept_sample = random.sample(kept_texts, min(len(kept_texts), 500))
    val_sample = random.sample(val_texts, min(len(val_texts), 500))
    for kt in kept_sample:
        for vt in val_sample:
            d = lev(kt, vt)
            if d <= 1:
                out.append((kt, vt, d))
                if len(out) >= sample_size:
                    return out
    return out


def load_vocab(vocab_path: Path) -> set[str]:
    data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return set(data)
    if isinstance(data, dict):
        return set(data.keys())
    raise ValueError(f"unsupported vocab format in {vocab_path}")


def _crop_word(page_path: Path, x: int, y: int, w: int, h: int, out_path: Path, target_h: int = 32) -> None:
    with Image.open(page_path) as im:
        _crop_from_image(im, x, y, w, h, out_path, target_h)


def _crop_from_image(
    im: "Image.Image", x: int, y: int, w: int, h: int, out_path: Path, target_h: int = 32
) -> None:
    W, H = im.size
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, W)
    y2 = min(y + h, H)
    crop = im.crop((x1, y1, x2, y2))
    if crop.height != target_h:
        ratio = target_h / max(crop.height, 1)
        new_w = max(1, int(crop.width * ratio))
        crop = crop.resize((new_w, target_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path, format="PNG")


def emit_labels_and_crops(
    conn: sqlite3.Connection,
    normalized_dir: Path,
    auto_val_fraction: float = 0.10,
) -> dict[str, int]:
    crops_dir = normalized_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    page_paths = dict(conn.execute("SELECT id, local_path FROM pages"))

    kept_rows = list(
        conn.execute(
            "SELECT w.id, w.page_id, w.x, w.y, w.w, w.h, w.text_nfc, p.source"
            " FROM words w JOIN pages p ON p.id=w.page_id WHERE w.status='kept' ORDER BY w.id"
        )
    )
    qa_rows = list(
        conn.execute(
            "SELECT w.id, w.page_id, w.x, w.y, w.w, w.h, w.text_nfc, p.source"
            " FROM words w JOIN pages p ON p.id=w.page_id WHERE w.status='exported_qa_candidate' ORDER BY w.id"
        )
    )

    import random
    val_pick: set[int] = set()
    if auto_val_fraction > 0:
        by_source: dict[str, list[int]] = {}
        for wid, _, _, _, _, _, _, src in kept_rows:
            by_source.setdefault(src, []).append(wid)
        for src, ids in by_source.items():
            k = max(1, int(len(ids) * auto_val_fraction))
            val_pick.update(random.sample(ids, min(k, len(ids))))

    train_lines: list[str] = []
    val_lines: list[str] = []
    qa_lines: list[str] = []

    # Keep at most one decoded page image resident at a time. Rows are ordered
    # by w.id which mirrors page-insertion order from the labeler, so consecutive
    # rows almost always share pid -- a single-slot cache hits ~100%.
    cur_pid: int | None = None
    cur_img: "Image.Image | None" = None

    def _ensure_page(pid: int) -> "Image.Image":
        nonlocal cur_pid, cur_img
        if pid != cur_pid:
            if cur_img is not None:
                cur_img.close()
            new_img = Image.open(page_paths[pid])
            new_img.load()
            cur_img = new_img
            cur_pid = pid
        assert cur_img is not None
        return cur_img

    try:
        for wid, pid, x, y, w, h, text, _src in kept_rows:
            out = crops_dir / f"{wid}.png"
            _crop_from_image(_ensure_page(pid), x, y, w, h, out)
            line = f"{out}|{text}"
            if wid in val_pick:
                val_lines.append(line)
            else:
                train_lines.append(line)

        for wid, pid, x, y, w, h, text, _src in qa_rows:
            out = crops_dir / f"{wid}.png"
            _crop_from_image(_ensure_page(pid), x, y, w, h, out)
            qa_lines.append(f"{out}|{text}|{wid}")
    finally:
        if cur_img is not None:
            cur_img.close()

    from scripts.data.auto_labeled_common import atomic_write_text
    atomic_write_text(normalized_dir / "train_labels.txt", "\n".join(train_lines) + ("\n" if train_lines else ""))
    atomic_write_text(normalized_dir / "val_labels.txt", "\n".join(val_lines) + ("\n" if val_lines else ""))
    atomic_write_text(normalized_dir / "qa_candidates.txt", "\n".join(qa_lines) + ("\n" if qa_lines else ""))

    return {
        "train": len(train_lines),
        "val": len(val_lines),
        "qa_candidates": len(qa_lines),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Filter labeled words and emit normalized label files.")
    p.add_argument("--db", type=Path, default=Path("data/external/auto_labeled/work.db"))
    p.add_argument("--min-conf", type=float, default=0.85)
    p.add_argument("--vocab", type=Path, default=Path("data/vocab.json"))
    p.add_argument("--existing-val", type=Path, nargs="*", default=[Path("data/combined/val_labels.txt")])
    p.add_argument("--existing-test", type=Path, nargs="*", default=[Path("data/combined/test_labels.txt")])
    p.add_argument("--existing-benchmarks", type=Path, nargs="*", default=[])
    p.add_argument(
        "--normalized-dir",
        type=Path,
        default=Path("data/external/auto_labeled/normalized"),
    )
    p.add_argument("--manifest", type=Path, default=Path("data/external/auto_labeled/manifests/manifest.json"))
    p.add_argument("--qa-per-source", type=int, default=500)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    conn = open_work_db(args.db)
    try:
        vocab = load_vocab(args.vocab)
        filter_confidence(conn, min_conf=args.min_conf)
        filter_oov(conn, vocab=vocab)
        filter_bbox(conn)
        filter_dup(conn)
        apply_text_disjoint_rule_1(
            conn,
            existing_val=args.existing_val,
            existing_test=args.existing_test,
            existing_benchmarks=args.existing_benchmarks,
        )
        promote_labeled_to_kept(conn)
        select_qa_candidates(conn, target_per_source=args.qa_per_source)
        affected = symmetric_qa_eviction(conn)
        counts = audit_disjointness(conn)
        if any(v != 0 for v in counts.values()) or affected != 0:
            print(f"AUDIT FAILED: {counts} symmetric_evicted={affected}", file=sys.stderr)
            conn.rollback()
            return 3
        emitted = emit_labels_and_crops(conn, normalized_dir=args.normalized_dir)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(
                {
                    "emitted": emitted,
                    "audit_counts": counts,
                    "min_conf": args.min_conf,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        conn.commit()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
