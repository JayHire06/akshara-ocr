#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.benchmark_suites import build_benchmark_suites


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build benchmark suites for synthetic-only, mixed, and real-only OCR evaluation."
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "data" / "benchmarks"),
        help="Directory where benchmark suites and manifest.json will be written.",
    )
    parser.add_argument(
        "--synthetic-labels",
        nargs="*",
        default=[],
        help="Pipe-separated labels files for clean synthetic samples.",
    )
    parser.add_argument(
        "--morphed-labels",
        nargs="*",
        default=[],
        help="Pipe-separated labels files for morphed or heavily augmented synthetic samples.",
    )
    parser.add_argument(
        "--real-labels",
        nargs="*",
        default=[],
        help="Pipe-separated labels files for real photographed text samples.",
    )
    parser.add_argument(
        "--iiit-train-labels",
        help="Optional IIIT train labels file. If provided, a holdout split will be created for testing.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--synthetic-size",
        type=int,
        default=None,
        help="Optional cap for the synthetic-only suite size.",
    )
    parser.add_argument(
        "--mixed-size",
        type=int,
        default=None,
        help="Optional total size for the mixed suite. It will be split 50/50 between morphed and real.",
    )
    parser.add_argument(
        "--real-size",
        type=int,
        default=None,
        help="Optional cap for the real-only suite size.",
    )
    parser.add_argument(
        "--iiit-holdout-ratio",
        type=float,
        default=0.15,
        help="Holdout ratio used when carving a benchmark split from the IIIT training labels.",
    )
    parser.add_argument(
        "--iiit-holdout-count",
        type=int,
        default=None,
        help="Override the IIIT holdout size with an exact count.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_benchmark_suites(
        output_dir=args.output_dir,
        synthetic_labels=args.synthetic_labels,
        morphed_labels=args.morphed_labels,
        real_labels=args.real_labels,
        iiit_train_labels=args.iiit_train_labels,
        seed=args.seed,
        synthetic_size=args.synthetic_size,
        mixed_size=args.mixed_size,
        real_size=args.real_size,
        iiit_holdout_ratio=args.iiit_holdout_ratio,
        iiit_holdout_count=args.iiit_holdout_count,
    )

    print(f"Benchmark suites written to: {Path(args.output_dir).resolve()}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
