from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from scripts.data.auto_labeled_common import open_work_db
from scripts.data.filter_and_crop_words import (
    emit_labels_and_crops,
    promote_labeled_to_kept,
)


def test_emit_writes_crops_and_label_files(tmp_path):
    page_image = tmp_path / "page.png"
    Image.new("L", (200, 100), 255).save(page_image)

    conn = open_work_db(tmp_path / "w.db")
    conn.execute(
        "INSERT INTO pages (id, source, uri, local_path, fetched_at, status)"
        " VALUES (1, 'wikipedia', 'u', ?, '2026-04-16', 'labeled')",
        (str(page_image),),
    )
    conn.execute(
        "INSERT INTO words (page_id, x, y, w, h, text_nfc, conf, provider, status)"
        " VALUES (1, 10, 20, 50, 30, 'alpha', 0.99, 'fake', 'kept')"
    )
    conn.execute(
        "INSERT INTO words (page_id, x, y, w, h, text_nfc, conf, provider, status)"
        " VALUES (1, 80, 20, 50, 30, 'beta', 0.99, 'fake', 'exported_qa_candidate')"
    )
    conn.commit()

    out_dir = tmp_path / "normalized"
    emit_labels_and_crops(
        conn,
        normalized_dir=out_dir,
        auto_val_fraction=0.0,
    )

    train_file = out_dir / "train_labels.txt"
    qa_file = out_dir / "qa_candidates.txt"
    assert train_file.exists()
    assert qa_file.exists()

    train_lines = train_file.read_text(encoding="utf-8").strip().splitlines()
    qa_lines = qa_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(train_lines) == 1
    assert len(qa_lines) == 1
    assert train_lines[0].endswith("|alpha")
    # QA lines carry a trailing |wid for write-back in the QA tool.
    assert "|beta|" in qa_lines[0]
    assert qa_lines[0].split("|")[-1].isdigit()

    crops = list((out_dir / "crops").glob("*.png"))
    assert len(crops) == 2
