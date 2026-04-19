"""Backfill the trackio `akshara-ocr` project with historical v1–v8 runs.

Why this exists:
    v9's training loop logs to trackio live, but every prior version
    (v1–v8) was trained before trackio was wired in. The leaderboard
    dashboard therefore shows v9 runs alone. This script walks each
    version's `best.pth` checkpoint and publishes enough data to let
    all versions share the same project view.

What gets published, per version:
    1. A replay of the per-epoch `metrics.jsonl` alongside the
       checkpoint, if one exists (only v5 has this today). Metrics are
       logged at their original epoch step under keys like
       `train/loss`, `val/loss`, `val/crr`, `lr`, `epoch_time_sec`.
    2. A single terminal datapoint with the measured CER/WER on
       `data/combined/val_labels.txt` (the canonical val all versions
       were trained against), logged at step 9999 so it doesn't visually
       merge with the replayed training curve.

Vocab handling:
    Older checkpoints have output head size 52 (pre-auto 52-char
    combined vocab). v9's new runs have 71 (70-char master + blank).
    `_load_vocab_list` in benchmark_eval only looks at the current
    `data/combined/vocab.json`, which is now the 70-char master — using
    it to decode a 52-output checkpoint gives garbage. This script picks
    the matching vocab file by checkpoint output size instead.

Architectures skipped:
    Some ultra-legacy checkpoints (v1 "Base Gen.", v3 "200K Extended")
    use the Sequential-CRNN format that `_load_model` explicitly
    refuses. Those are reported as `SKIPPED(arch)` in the summary and
    do not create a trackio run.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import trackio  # noqa: E402

from model.benchmark_eval import (  # noqa: E402
    CheckpointLoadError,
    _load_model,
    calculate_cer,
    calculate_wer,
    discover_default_checkpoints,
)
from model.ctc_decoder import CTCDecoder  # noqa: E402


# Maps output-head size → vocab file that was in effect when that size
# was trained. Pre-auto combined had 52 entries (including <blank>);
# the master vocab has 70. The 52-char vocab lives at the repo root
# (`./vocab.json`, tracked) as the canonical source; the data/combined/
# backup (`vocab.pre_auto.bak.json`) is an untracked convenience copy
# we prefer when present but don't require.
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


def _vocab_list_for_size(vocab_size: int) -> list[str]:
    """Pick a vocab file matching the checkpoint's head width.

    Tries each candidate path in order; falls back to the current
    combined vocab if nothing matches (results will be garbage if the
    sizes disagree, but we'd rather log a bogus number than crash).
    """
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


def _eval_checkpoint(ckpt_path: Path, labels_path: Path, device: torch.device) -> dict:
    """Run greedy-CTC eval of one checkpoint on one labels file.

    Returns a dict with cer, wer, samples. Reproduces benchmark_eval's
    preprocessing (grayscale, h=32, /255) and batch structure but carries
    its own vocab-list resolution so old checkpoints decode correctly.
    """
    model, vocab_size = _load_model(ckpt_path, device)
    vocab_list = _vocab_list_for_size(vocab_size)
    decoder = CTCDecoder(vocab=vocab_list, blank_id=0)

    total_cer = total_wer = 0.0
    n = 0
    batch_imgs: list[np.ndarray] = []
    batch_gts: list[str] = []

    def _flush() -> None:
        nonlocal total_cer, total_wer, n
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
            total_cer += calculate_cer(pred, gt)
            total_wer += calculate_wer(pred, gt)
            n += 1

    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|" not in line:
                continue
            path_s, gt = line.split("|", 1)
            ipath = Path(path_s)
            if not ipath.exists():
                continue
            img = Image.open(ipath).convert("L")
            w, h = img.size
            img = img.resize((max(1, int(w * 32 / max(h, 1))), 32))
            batch_imgs.append(np.asarray(img, dtype=np.float32) / 255.0)
            batch_gts.append(gt)
            if len(batch_imgs) == 128:
                _flush()
                batch_imgs.clear()
                batch_gts.clear()

    if batch_imgs:
        _flush()

    if n == 0:
        return {"cer_pct": None, "wer_pct": None, "samples": 0}
    return {"cer_pct": total_cer / n * 100, "wer_pct": total_wer / n * 100, "samples": n}


def _run_name_for(label: str) -> str:
    # e.g. "v1 (Base Gen.)" → "historical_v1"
    return "historical_" + label.split(" ")[0].lower()


def _find_metrics_file(ckpt_path: Path) -> Path | None:
    metrics = ckpt_path.parent / "metrics.jsonl"
    return metrics if metrics.exists() else None


def _replay_metrics(metrics_file: Path) -> int:
    """Stream a historical metrics.jsonl into the current trackio run.

    Handles the v5 schema (train_loss, val_loss, val_crr, lr, epoch_time).
    Silently skips lines that fail to parse.
    """
    count = 0
    with open(metrics_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = {}
            if "train_loss" in m:
                payload["train/loss"] = m["train_loss"]
            if "val_loss" in m:
                payload["val/loss"] = m["val_loss"]
            if "val_crr" in m:
                # Convert CRR (character recognition rate, %) to CER proxy.
                payload["val/crr_pct"] = m["val_crr"]
                payload["val/cer_pct_from_crr"] = 100.0 - m["val_crr"]
            if "lr" in m:
                payload["lr"] = m["lr"]
            if "epoch_time" in m:
                payload["epoch_time_sec"] = m["epoch_time"]
            if not payload:
                continue
            trackio.log(payload, step=int(m.get("epoch", count)))
            count += 1
    return count


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench_labels = PROJECT_ROOT / "data/combined/val_labels.txt"
    if not bench_labels.exists():
        print(f"error: benchmark labels not found at {bench_labels}")
        return 1

    print(f"Backfilling historical runs into trackio project 'akshara-ocr'")
    print(f"Device: {device}")
    print(f"Benchmark: {bench_labels} ({sum(1 for _ in open(bench_labels))} rows)")
    print()

    summary: list[tuple[str, str]] = []

    # Skip v9 — it already logs to trackio live, don't double-count.
    versions = [(l, p) for l, p in discover_default_checkpoints(PROJECT_ROOT) if "v9" not in l.lower()]

    for label, ckpt_path in versions:
        if ckpt_path is None:
            print(f"[skip] {label}: no checkpoint discovered")
            summary.append((label, "NO_CHECKPOINT"))
            continue

        ckpt_path = Path(ckpt_path)
        metrics_file = _find_metrics_file(ckpt_path)
        run_name = _run_name_for(label)

        print(f"=== {label} ===")
        print(f"  checkpoint: {ckpt_path.relative_to(PROJECT_ROOT)}")
        print(f"  metrics.jsonl: {'yes' if metrics_file else 'no'}")

        # Probe the checkpoint first; if it refuses to load, skip trackio init.
        try:
            probe_start = time.time()
            result = _eval_checkpoint(ckpt_path, bench_labels, device)
            probe_dt = time.time() - probe_start
        except CheckpointLoadError as exc:
            reason = str(exc).split(":", 1)[1].strip().split(".")[0] if ":" in str(exc) else str(exc)
            print(f"  SKIP (load error): {reason[:120]}")
            summary.append((label, f"SKIPPED({reason[:60]})"))
            continue
        except Exception as exc:
            print(f"  SKIP (eval error): {type(exc).__name__}: {exc}")
            summary.append((label, f"ERROR({type(exc).__name__})"))
            traceback.print_exc()
            continue

        # Good to go. Open a trackio run.
        trackio.init(
            project="akshara-ocr",
            name=run_name,
            config={
                "arch": label,
                "checkpoint": str(ckpt_path),
                "historical_backfill": True,
            },
        )

        replayed = 0
        if metrics_file is not None:
            try:
                replayed = _replay_metrics(metrics_file)
                print(f"  replayed {replayed} metrics rows from {metrics_file.name}")
            except Exception as exc:
                print(f"  warn: metrics replay failed: {exc}")

        trackio.log(
            {
                "final/cer_pct": result["cer_pct"],
                "final/wer_pct": result["wer_pct"],
                "final/samples": result["samples"],
            },
            step=9999,
        )
        trackio.finish()

        cer = f"{result['cer_pct']:.2f}%" if result["cer_pct"] is not None else "—"
        wer = f"{result['wer_pct']:.2f}%" if result["wer_pct"] is not None else "—"
        print(f"  final: CER {cer}  WER {wer}  n={result['samples']}  ({probe_dt:.1f}s)")
        summary.append(
            (label, f"OK cer={cer} wer={wer} replayed={replayed}")
        )

    print()
    print("=" * 64)
    print("Summary:")
    for label, status in summary:
        print(f"  {label:30s}  {status}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
