#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.v8_dataset_prep import merge_label_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge normalized dataset labels into a named v8 training stage."
    )
    parser.add_argument("--stage-name", required=True, help="Training stage name, e.g. stage1_base.")
    parser.add_argument(
        "--output-root",
        default=str(ROOT_DIR / "data" / "v8" / "stages"),
        help="Root directory where the stage folder will be created.",
    )
    parser.add_argument(
        "--train-labels",
        nargs="+",
        required=True,
        help="One or more normalized train label files to merge.",
    )
    parser.add_argument(
        "--val-labels",
        nargs="+",
        required=True,
        help="One or more normalized validation label files to merge.",
    )
    parser.add_argument(
        "--notes",
        nargs="*",
        default=[],
        help="Optional free-form notes stored in the stage manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage_dir = Path(args.output_root) / args.stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)

    train_count, train_summary = merge_label_files(
        label_files=args.train_labels,
        output_file=stage_dir / "train_labels.txt",
        source_name=f"{args.stage_name}:train",
    )
    val_count, val_summary = merge_label_files(
        label_files=args.val_labels,
        output_file=stage_dir / "val_labels.txt",
        source_name=f"{args.stage_name}:val",
    )

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "stage_name": args.stage_name,
        "train_count": train_count,
        "val_count": val_count,
        "train_summary": train_summary,
        "val_summary": val_summary,
        "notes": args.notes,
    }

    manifest_path = stage_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"Stage written to: {stage_dir.resolve()}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
