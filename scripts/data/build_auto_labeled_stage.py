from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def build_stage(
    stage_name: str,
    train_labels: Path,
    val_labels: Path,
    out_dir: Path,
    notes: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(train_labels, out_dir / "train_labels.txt")
    shutil.copyfile(val_labels, out_dir / "val_labels.txt")
    manifest = {
        "stage_name": stage_name,
        "train_labels": str((out_dir / "train_labels.txt").resolve()),
        "val_labels": str((out_dir / "val_labels.txt").resolve()),
        "notes": notes,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a v8-compatible stage from auto-labeled outputs.")
    p.add_argument("--stage-name", default="stage_auto_labeled_printed")
    p.add_argument(
        "--train-labels",
        type=Path,
        default=Path("data/external/auto_labeled/normalized/train_labels.txt"),
    )
    p.add_argument(
        "--val-labels",
        type=Path,
        default=Path("data/external/auto_labeled/normalized/val_labels.txt"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/v8/stages/stage_auto_labeled_printed"),
    )
    p.add_argument("--notes", default="Auto-labeled Tier-1 + Tier-4 printed Hindi")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_stage(
        stage_name=args.stage_name,
        train_labels=args.train_labels,
        val_labels=args.val_labels,
        out_dir=args.out_dir,
        notes=args.notes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
