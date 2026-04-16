from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.data.auto_labeled_common import open_work_db

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


@pytest.fixture
def vocab_file(tmp_path) -> Path:
    v = tmp_path / "vocab.json"
    chars = set("भारतहिंदीविकिपडयकहलो")
    v.write_text(json.dumps(sorted(chars)), encoding="utf-8")
    return v


def test_end_to_end_with_fake_provider(tmp_path, vocab_file):
    db_path = tmp_path / "work.db"
    raw_dir = tmp_path / "raw" / "wikipedia"
    raw_dir.mkdir(parents=True)

    fake_page = raw_dir / "page_wiki_001.png"
    from PIL import Image
    Image.new("L", (300, 100), 200).save(fake_page)

    conn = open_work_db(db_path)
    conn.execute(
        "INSERT INTO pages (source, uri, local_path, fetched_at, status)"
        " VALUES ('wikipedia', 'u', ?, '2026-04-16', 'pending')",
        (str(fake_page),),
    )
    conn.commit()
    conn.close()

    fake_json = Path("tests/fixtures/auto_labeled/fake_responses.json").resolve()

    _run([
        sys.executable,
        "scripts/data/label_with_cloud_ocr.py",
        "--db", str(db_path),
        "--provider", "fake",
        "--fake-responses", str(fake_json),
        "--concurrency", "1",
    ])

    normalized = tmp_path / "normalized"
    manifest = tmp_path / "manifest.json"
    _run([
        sys.executable,
        "scripts/data/filter_and_crop_words.py",
        "--db", str(db_path),
        "--vocab", str(vocab_file),
        "--existing-val",
        "--existing-test",
        "--normalized-dir", str(normalized),
        "--manifest", str(manifest),
        "--qa-per-source", "1",
    ])

    train = normalized / "train_labels.txt"
    assert train.exists()
    assert len(train.read_text(encoding="utf-8").strip().splitlines()) >= 1

    m = json.loads(manifest.read_text(encoding="utf-8"))
    assert m["audit_counts"]["kept_vs_existing_val"] == 0
