"""Cross-version × cross-benchmark evaluation matrix.

Runs every discovered model checkpoint against every configured benchmark
and emits:
  - A formatted markdown report (to stdout).
  - A JSON file (at --json-out) with the full (model, benchmark) → metrics
    grid, usable by downstream tooling.

Metrics per (model, benchmark):
  - CER (character error rate, %)
  - WER (word error rate, %)
  - Exact-match rate (%)
  - Empty-prediction rate (%)
  - Samples evaluated (n)
  - Inference throughput (samples/sec)

Vocab resolution follows the same rule as `backfill_trackio.py`: pick the
vocab file whose dict size matches the checkpoint's output-head size, so
legacy 52-output checkpoints decode under the pre-auto vocab instead of
the current 70-char master.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from model.benchmark_eval import (  # noqa: E402
    CheckpointLoadError,
    _load_model,
    calculate_cer,
    calculate_wer,
    discover_default_checkpoints,
    normalize_text,
)
from model.ctc_decoder import CTCDecoder  # noqa: E402


VOCAB_FILES = {
    52: [
        PROJECT_ROOT / "data/combined/vocab.pre_auto.bak.json",
        PROJECT_ROOT / "vocab.json",
    ],
    70: [
        PROJECT_ROOT / "data/combined/vocab.json",
        PROJECT_ROOT / "data/vocab.json",
    ],
}

BENCHMARKS = [
    (
        "combined_val",
        PROJECT_ROOT / "data/combined/val_labels.txt",
        "Text-disjoint combined val — the canonical historical benchmark.",
    ),
    (
        "verified_test",
        PROJECT_ROOT / "data/external/auto_labeled/normalized/verified_test_labels.txt",
        "Auto-labeled confidence-gated test split (Azure conf ≥ 0.95).",
    ),
    (
        "auto_labeled_heldout_val",
        PROJECT_ROOT / "data/external/auto_labeled/normalized/held_out_val_labels.txt",
        "Held-out half of the auto-labeled val split (never used for training or QA gating).",
    ),
]

BATCH_SIZE = 128


def _vocab_list_for_size(vocab_size: int) -> list[str]:
    candidates = list(VOCAB_FILES.get(vocab_size, []))
    candidates.append(PROJECT_ROOT / "data/combined/vocab.json")
    for path in candidates:
        if path.exists():
            vocab_dict = json.load(open(path, encoding="utf-8"))
            out = ["<UNK>"] * vocab_size
            for char, idx in vocab_dict.items():
                if 0 <= idx < vocab_size:
                    out[idx] = "<PAD>" if char == "<blank>" else char
            return out
    return ["<UNK>"] * vocab_size


def _prep_image(path: Path) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("L")
    except Exception:
        return None
    w, h = img.size
    nw = max(1, int(w * 32 / max(h, 1)))
    img = img.resize((nw, 32))
    return np.asarray(img, dtype=np.float32) / 255.0


def _evaluate(
    model: torch.nn.Module, decoder: CTCDecoder, labels_path: Path, device: torch.device
) -> dict:
    total_cer = total_wer = 0.0
    exact = empty = n = 0
    t0 = time.time()

    batch_imgs: list[np.ndarray] = []
    batch_gts: list[str] = []

    def _flush() -> None:
        nonlocal total_cer, total_wer, exact, empty, n
        max_w = max(a.shape[1] for a in batch_imgs)
        padded = np.stack(
            [
                np.pad(a, ((0, 0), (0, max_w - a.shape[1])), constant_values=0)
                for a in batch_imgs
            ]
        )
        tensor = torch.tensor(padded).unsqueeze(1).to(device)
        with torch.no_grad():
            out = model(tensor)
        preds = decoder.decode_best_path(out)
        for pred, gt in zip(preds, batch_gts):
            pred = pred.replace("<UNK>", "").strip()
            if pred == "":
                empty += 1
            total_cer += calculate_cer(pred, gt)
            total_wer += calculate_wer(pred, gt)
            if normalize_text(pred) == normalize_text(gt):
                exact += 1
            n += 1

    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|" not in line:
                continue
            path_s, gt = line.split("|", 1)
            arr = _prep_image(Path(path_s))
            if arr is None:
                continue
            batch_imgs.append(arr)
            batch_gts.append(gt)
            if len(batch_imgs) >= BATCH_SIZE:
                _flush()
                batch_imgs.clear()
                batch_gts.clear()

    if batch_imgs:
        _flush()

    dt = time.time() - t0
    if n == 0:
        return {
            "cer_pct": None,
            "wer_pct": None,
            "exact_pct": None,
            "empty_pct": None,
            "samples": 0,
            "samples_per_sec": None,
        }
    return {
        "cer_pct": round(total_cer / n * 100, 3),
        "wer_pct": round(total_wer / n * 100, 3),
        "exact_pct": round(exact / n * 100, 3),
        "empty_pct": round(empty / n * 100, 3),
        "samples": n,
        "samples_per_sec": round(n / dt, 1) if dt > 0 else None,
    }


def _cell(metrics: dict | None, key: str) -> str:
    if metrics is None:
        return "—"
    v = metrics.get(key)
    if v is None:
        return "—"
    return f"{v:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PROJECT_ROOT / "docs/benchmark-matrix.json",
        help="Where to write the full grid as JSON.",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="Optional: write the markdown report to this file in addition to stdout.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"# Akshara-OCR benchmark matrix", file=sys.stderr)
    print(f"device={device}", file=sys.stderr)

    # Confirm all benchmarks exist up-front.
    benchmarks = [(k, p, d) for k, p, d in BENCHMARKS if p.exists()]
    for k, p, _ in BENCHMARKS:
        if not p.exists():
            print(f"  WARN: benchmark {k} not found at {p} — skipping", file=sys.stderr)

    # Pre-count sample counts so we can report them.
    bench_meta = {}
    for k, p, d in benchmarks:
        n_rows = sum(1 for line in open(p, encoding="utf-8") if "|" in line)
        bench_meta[k] = {"path": str(p.relative_to(PROJECT_ROOT)), "rows": n_rows, "desc": d}

    versions = discover_default_checkpoints(PROJECT_ROOT)

    # Grid: {version_label: {benchmark_key: metrics_dict | None}}
    grid: dict[str, dict[str, dict | None]] = {}
    arch_info: dict[str, dict] = {}

    for label, ckpt_path in versions:
        if ckpt_path is None:
            print(f"[skip] {label}: no checkpoint discovered", file=sys.stderr)
            grid[label] = {k: None for k, _, _ in benchmarks}
            arch_info[label] = {"checkpoint": None, "status": "NO_CHECKPOINT"}
            continue

        print(f"[{label}] loading {Path(ckpt_path).relative_to(PROJECT_ROOT)}", file=sys.stderr)
        try:
            model, vocab_size = _load_model(ckpt_path, device)
        except CheckpointLoadError as exc:
            reason = str(exc).split(":", 1)[-1].strip().split(".")[0][:80]
            print(f"  SKIP (load error): {reason}", file=sys.stderr)
            grid[label] = {k: None for k, _, _ in benchmarks}
            arch_info[label] = {"checkpoint": str(ckpt_path), "status": f"SKIPPED({reason})"}
            continue
        except Exception as exc:
            print(f"  SKIP (unexpected): {type(exc).__name__}: {exc}", file=sys.stderr)
            grid[label] = {k: None for k, _, _ in benchmarks}
            arch_info[label] = {"checkpoint": str(ckpt_path), "status": f"ERROR({type(exc).__name__})"}
            continue

        decoder = CTCDecoder(vocab=_vocab_list_for_size(vocab_size), blank_id=0)
        arch_info[label] = {
            "checkpoint": str(Path(ckpt_path).relative_to(PROJECT_ROOT)),
            "vocab_size": vocab_size,
            "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "status": "OK",
        }

        grid[label] = {}
        for key, path, _ in benchmarks:
            metrics = _evaluate(model, decoder, path, device)
            grid[label][key] = metrics
            tag = (
                f"cer={metrics['cer_pct']:.2f}% wer={metrics['wer_pct']:.2f}% "
                f"exact={metrics['exact_pct']:.2f}% n={metrics['samples']}"
                if metrics["samples"] else "empty"
            )
            print(f"  {key:<26s} {tag}", file=sys.stderr)

        # Free model memory before moving to the next one
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Emit JSON
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "device": str(device),
                "benchmarks": bench_meta,
                "arch": arch_info,
                "grid": grid,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nJSON written to {args.json_out.relative_to(PROJECT_ROOT)}", file=sys.stderr)

    # Emit markdown
    md_lines: list[str] = []
    md_lines.append("# Benchmark matrix — Akshara OCR")
    md_lines.append("")
    md_lines.append(
        f"_Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}, device `{device}`._"
    )
    md_lines.append("")
    md_lines.append("## Benchmarks")
    md_lines.append("")
    md_lines.append("| Key | Rows | Description |")
    md_lines.append("|---|---:|---|")
    for k, meta in bench_meta.items():
        md_lines.append(f"| `{k}` | {meta['rows']:,} | {meta['desc']} |")
    md_lines.append("")

    # One table per metric for readability
    for metric_key, label in [
        ("cer_pct", "CER (Character Error Rate, %)"),
        ("wer_pct", "WER (Word Error Rate, %)"),
        ("exact_pct", "Exact-match (%)"),
        ("empty_pct", "Empty-prediction rate (%)"),
    ]:
        md_lines.append(f"## {label}")
        md_lines.append("")
        header = "| Model | " + " | ".join(f"`{k}`" for k, _, _ in benchmarks) + " |"
        sep = "|---|" + "---:|" * len(benchmarks)
        md_lines.append(header)
        md_lines.append(sep)
        for label_v, per_bench in grid.items():
            cells = [_cell(per_bench.get(k), metric_key) for k, _, _ in benchmarks]
            md_lines.append(f"| {label_v} | " + " | ".join(cells) + " |")
        md_lines.append("")

    md_lines.append("## Architecture & load status")
    md_lines.append("")
    md_lines.append("| Model | Checkpoint | Vocab size | Params (M) | Status |")
    md_lines.append("|---|---|---:|---:|---|")
    for label_v, info in arch_info.items():
        md_lines.append(
            f"| {label_v} | `{info.get('checkpoint', '—')}` | {info.get('vocab_size', '—')} | "
            f"{info.get('params_M', '—')} | {info['status']} |"
        )

    md = "\n".join(md_lines) + "\n"
    print(md)
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(md, encoding="utf-8")
        print(f"Markdown written to {args.md_out.relative_to(PROJECT_ROOT)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
