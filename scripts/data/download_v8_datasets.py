#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.v8_dataset_prep import parse_tab_gt, write_normalized_dataset


DATASET_CATALOG = {
    "mozhi_hindi": {
        "dataset_name": "Mozhi-Hindi",
        "dataset_slug": "mozhi_hindi",
        "language": "hi",
        "modality": "printed",
        "source_url": "https://ilocr.iiit.ac.in/dataset/7/",
        "base_url": "https://cdn.iiit.ac.in/cdn/ilocr.iiit.ac.in/public/printed/phase-0/v0.5/hindi/akshara",
        "splits": {
            "train": {"archive": "train.zip", "gt_file": "train_gt.txt"},
            "val": {"archive": "val.zip", "gt_file": "val_gt.txt"},
            "test": {"archive": "test.zip", "gt_file": "test_gt.txt"},
        },
    },
    "iiit_hw_words_hindi": {
        "dataset_name": "IIIT-INDIC-HW-WORDS-Hindi",
        "dataset_slug": "iiit_hw_words_hindi",
        "language": "hi",
        "modality": "handwritten",
        "source_url": "https://ilocr.iiit.ac.in/dataset/18/",
        "base_url": "https://cdn.iiit.ac.in/cdn/ilocr.iiit.ac.in/public/handwritten/phase-0/v0.5/hindi/IIIT-INDIC-HW-WORDS-Devanagari",
        "splits": {
            "train": {"archive": "train.zip", "gt_file": "train_gt.txt"},
            "val": {"archive": "val.zip", "gt_file": "val_gt.txt"},
            "test": {"archive": "test.zip", "gt_file": "test_gt.txt"},
        },
    },
    "iiit_synth_r_hindi": {
        "dataset_name": "IIIT-Synthetic-R-Hindi",
        "dataset_slug": "iiit_synth_r_hindi",
        "language": "hi",
        "modality": "printed_synthetic",
        "source_url": "https://ilocr.iiit.ac.in/dataset/49/",
        "base_url": "https://cdn.iiit.ac.in/cdn/ilocr.iiit.ac.in/public/printed/phase-0/v0.5/hindi/IIIT-Synthetic-R-Hindi",
        "splits": {
            "train": {"archive": "train.zip", "gt_file": "train_gt.txt"},
            "val": {"archive": "val.zip", "gt_file": "val_gt.txt"},
            "test": {"archive": "test.zip", "gt_file": "test_gt.txt"},
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and normalize selected v8 datasets into data/external."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_CATALOG.keys()),
        required=True,
        help="Datasets to download and normalize.",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT_DIR / "data" / "external"),
        help="Root directory for external datasets.",
    )
    parser.add_argument(
        "--inspect-limit",
        type=int,
        default=1000,
        help="Maximum images to inspect per split while building manifests.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip network download and only normalize previously extracted archives.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[skip] archive already exists: {destination}")
        return

    print(f"[download] {url}")
    with urllib.request.urlopen(url) as response, open(destination, "wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    print(f"[saved] {destination}")


def extract_zip(archive_path: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".extracted"
    if marker.exists():
        print(f"[skip] already extracted: {extract_dir}")
        return

    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive_path} -> {extract_dir}")
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(extract_dir)
    marker.write_text("ok\n", encoding="utf-8")


def _resolve_images_dir(split_dir: Path) -> Path:
    direct = split_dir / "images"
    nested = split_dir / "train" / "images"
    if direct.exists():
        return direct
    if nested.exists():
        return nested

    matches = list(split_dir.rglob("images"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Unable to locate images directory under {split_dir}")


def _resolve_gt_file(split_dir: Path, expected_name: str) -> Path:
    direct = split_dir / expected_name
    if direct.exists():
        return direct

    matches = list(split_dir.rglob(expected_name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Unable to locate {expected_name} under {split_dir}")


def normalize_dataset(dataset_key: str, output_root: Path, inspect_limit: int) -> Path:
    spec = DATASET_CATALOG[dataset_key]
    dataset_root = output_root / spec["dataset_slug"]
    raw_root = dataset_root / "raw"

    split_samples = {}
    import_reports = {}
    source_inputs = {}

    for split_name, split_spec in spec["splits"].items():
        split_dir = raw_root / split_name
        gt_file = _resolve_gt_file(split_dir, split_spec["gt_file"])
        images_dir = _resolve_images_dir(split_dir)
        samples, report = parse_tab_gt(
            gt_path=gt_file,
            images_dir=images_dir,
            split=split_name,
            source_id=spec["dataset_slug"],
        )
        split_samples[split_name] = samples
        import_reports[split_name] = report
        source_inputs[split_name] = {
            "gt_file": str(gt_file.resolve()),
            "images_dir": str(images_dir.resolve()),
        }

    manifest_path = write_normalized_dataset(
        dataset_root=dataset_root,
        dataset_name=spec["dataset_name"],
        dataset_slug=spec["dataset_slug"],
        language=spec["language"],
        modality=spec["modality"],
        source_url=spec["source_url"],
        split_samples=split_samples,
        source_inputs=source_inputs,
        import_reports=import_reports,
        inspect_limit=inspect_limit,
        notes=[
            "Dataset downloaded from official NLTM / IIIT OCR distribution endpoints.",
            "Archives are preserved under raw/ and normalization outputs point to extracted image files.",
        ],
    )
    return manifest_path


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifests = {}
    for dataset_key in args.datasets:
        spec = DATASET_CATALOG[dataset_key]
        dataset_root = output_root / spec["dataset_slug"]
        raw_root = dataset_root / "raw"
        archives_root = raw_root / "archives"

        print(f"\n=== {spec['dataset_name']} ===")

        if not args.skip_download:
            for split_name, split_spec in spec["splits"].items():
                archive_name = split_spec["archive"]
                archive_url = f"{spec['base_url']}/{archive_name}"
                archive_path = archives_root / archive_name
                split_dir = raw_root / split_name
                download_file(archive_url, archive_path)
                extract_zip(archive_path, split_dir)

        manifest_path = normalize_dataset(dataset_key, output_root=output_root, inspect_limit=args.inspect_limit)
        manifests[dataset_key] = str(manifest_path.resolve())
        print(f"[normalized] manifest: {manifest_path}")

    print(json.dumps(manifests, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
