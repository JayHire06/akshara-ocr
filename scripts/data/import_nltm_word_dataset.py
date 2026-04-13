#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.v8_dataset_prep import parse_tab_gt, write_normalized_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize an NLTM/IIIT word dataset into the v8 external dataset layout."
    )
    parser.add_argument("--dataset-name", required=True, help="Human-readable dataset name.")
    parser.add_argument("--dataset-slug", required=True, help="Filesystem-safe dataset slug.")
    parser.add_argument("--language", default="hi", help="Language code.")
    parser.add_argument("--modality", required=True, help="Dataset modality, e.g. printed or handwritten.")
    parser.add_argument("--source-url", required=True, help="Primary dataset source URL.")
    parser.add_argument(
        "--output-root",
        default=str(ROOT_DIR / "data" / "external"),
        help="Root directory where the normalized dataset folder will be created.",
    )
    parser.add_argument("--train-images-dir", help="Directory containing train images.")
    parser.add_argument("--train-gt", help="Ground-truth text file for the train split.")
    parser.add_argument("--val-images-dir", help="Directory containing validation images.")
    parser.add_argument("--val-gt", help="Ground-truth text file for the validation split.")
    parser.add_argument("--test-images-dir", help="Directory containing test images.")
    parser.add_argument("--test-gt", help="Ground-truth text file for the test split.")
    parser.add_argument(
        "--inspect-limit",
        type=int,
        default=1000,
        help="Maximum images to inspect per split for QA stats.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.output_root) / args.dataset_slug

    split_inputs = {
        "train": (args.train_gt, args.train_images_dir),
        "val": (args.val_gt, args.val_images_dir),
        "test": (args.test_gt, args.test_images_dir),
    }

    split_samples = {}
    import_reports = {}
    source_inputs = {}

    for split_name, (gt_file, images_dir) in split_inputs.items():
        if not gt_file and not images_dir:
            continue
        if not gt_file or not images_dir:
            raise SystemExit(f"{split_name} split requires both --{split_name}-gt and --{split_name}-images-dir")

        samples, report = parse_tab_gt(
            gt_path=gt_file,
            images_dir=images_dir,
            split=split_name,
            source_id=args.dataset_slug,
        )
        split_samples[split_name] = samples
        import_reports[split_name] = report
        source_inputs[split_name] = {
            "gt_file": str(Path(gt_file).resolve()),
            "images_dir": str(Path(images_dir).resolve()),
        }

    if not split_samples:
        raise SystemExit("No dataset splits were supplied.")

    manifest_path = write_normalized_dataset(
        dataset_root=dataset_root,
        dataset_name=args.dataset_name,
        dataset_slug=args.dataset_slug,
        language=args.language,
        modality=args.modality,
        source_url=args.source_url,
        split_samples=split_samples,
        source_inputs=source_inputs,
        import_reports=import_reports,
        inspect_limit=args.inspect_limit,
        notes=[
            "Importer expects GT files where each line is `filename<TAB>ground_truth`.",
            "Raw images are preserved in place; normalized labels point to absolute image paths.",
        ],
    )

    print(f"Normalized dataset written to: {dataset_root.resolve()}")
    print(f"Manifest written to: {manifest_path.resolve()}")
    with open(manifest_path, encoding='utf-8') as handle:
        manifest = json.load(handle)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
