from __future__ import annotations

from pathlib import Path

from scripts.data.build_auto_labeled_stage import build_stage


def test_build_stage_produces_expected_files(tmp_path):
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    (normalized / "train_labels.txt").write_text("/a.png|alpha\n", encoding="utf-8")
    (normalized / "val_labels.txt").write_text("/b.png|beta\n", encoding="utf-8")

    stage_dir = tmp_path / "stages" / "stage_auto_labeled_printed"
    build_stage(
        stage_name="stage_auto_labeled_printed",
        train_labels=normalized / "train_labels.txt",
        val_labels=normalized / "val_labels.txt",
        out_dir=stage_dir,
        notes="test",
    )
    assert (stage_dir / "train_labels.txt").exists()
    assert (stage_dir / "val_labels.txt").exists()
    assert (stage_dir / "manifest.json").exists()
