#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from model.benchmark_eval import (
    CheckpointLoadError,
    discover_default_checkpoints,
    evaluate_checkpoint_on_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all historical checkpoints against named benchmark suites."
    )
    parser.add_argument(
        "--benchmark-root",
        default=str(ROOT_DIR / "data" / "benchmarks"),
        help="Directory containing manifest.json and benchmark labels files.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Optional subset of benchmark names to evaluate.",
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT_DIR / "tests" / "benchmarks" / "suite_report.json"),
        help="Path where the benchmark report JSON will be written.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=0,
        help="Include up to N sample predictions per checkpoint and benchmark in the JSON report.",
    )
    return parser.parse_args()


def load_benchmarks(benchmark_root: Path, names: set[str] | None) -> list[dict]:
    manifest_path = benchmark_root / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        benchmarks = manifest.get("benchmarks", [])
    else:
        benchmarks = []
        for labels_path in sorted(benchmark_root.glob("*/labels.txt")):
            benchmarks.append(
                {
                    "name": labels_path.parent.name,
                    "labels_file": str(labels_path.resolve()),
                    "sample_count": None,
                }
            )

    if names is None:
        return benchmarks
    return [benchmark for benchmark in benchmarks if benchmark["name"] in names]


def main() -> int:
    args = parse_args()
    benchmark_root = Path(args.benchmark_root)
    selected_names = set(args.benchmarks) if args.benchmarks else None
    benchmarks = load_benchmarks(benchmark_root, selected_names)

    if not benchmarks:
        raise SystemExit(f"No benchmarks found under {benchmark_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = discover_default_checkpoints(ROOT_DIR)

    report = {
        "benchmark_root": str(benchmark_root.resolve()),
        "device": str(device),
        "benchmarks": {},
    }

    print("======================================================")
    print("      Akshara-OCR Benchmark Suite Evaluation")
    print("======================================================")

    for benchmark in benchmarks:
        name = benchmark["name"]
        labels_file = Path(benchmark["labels_file"])
        print(f"\n[{name}] {labels_file}")
        print(f"{'Model Version':<25} | {'CER':<10} | {'WER':<10} | {'Empty':<10} | {'N':<6}")
        print("-" * 76)

        benchmark_results = {}
        for model_name, checkpoint_path in checkpoints:
            if checkpoint_path is None:
                benchmark_results[model_name] = {
                    "checkpoint": None,
                    "cer": None,
                    "wer": None,
                    "empty_rate": None,
                    "samples_evaluated": 0,
                }
                print(f"{model_name:<25} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {0:<6}")
                continue

            try:
                metrics = evaluate_checkpoint_on_labels(
                    checkpoint_path=checkpoint_path,
                    labels_file=labels_file,
                    device=device,
                    root_dir=ROOT_DIR,
                    preview_limit=args.preview_limit,
                )
                benchmark_results[model_name] = {"checkpoint": checkpoint_path, **metrics}

                cer = "N/A" if metrics["cer"] is None else f"{metrics['cer']:.2f}%"
                wer = "N/A" if metrics["wer"] is None else f"{metrics['wer']:.2f}%"
                empty_rate = (
                    "N/A" if metrics["empty_rate"] is None else f"{metrics['empty_rate']:.2f}%"
                )
                print(
                    f"{model_name:<25} | {cer:<10} | {wer:<10} | {empty_rate:<10} | "
                    f"{metrics['samples_evaluated']:<6}"
                )
            except CheckpointLoadError as err:
                benchmark_results[model_name] = {
                    "checkpoint": checkpoint_path,
                    "cer": None,
                    "wer": None,
                    "empty_rate": None,
                    "samples_evaluated": 0,
                    "error": str(err),
                }
                print(
                    f"{model_name:<25} | {'INCOMPAT':<10} | {'INCOMPAT':<10} | "
                    f"{'INCOMPAT':<10} | {0:<6}"
                )
                print(f"    ↳ {err}")

        report["benchmarks"][name] = {
            "labels_file": str(labels_file.resolve()),
            "results": benchmark_results,
        }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"\nReport saved to: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
