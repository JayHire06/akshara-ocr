#!/usr/bin/env python3
"""
Rebuild data/combined/train_labels.txt + val_labels.txt as a TEXT-DISJOINT split.

Every target string ends up in exactly one split. Guarantees zero overlap between
train and val target texts, so validation CER/WER actually measure generalization
instead of memorization.

Original files are backed up with a ``.leaky.bak`` suffix before overwrite.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
COMBINED = ROOT / "data" / "combined"
TRAIN_LABELS = COMBINED / "train_labels.txt"
VAL_LABELS = COMBINED / "val_labels.txt"

VAL_RATIO = 0.15
SEED = 20260413


def load_labels(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if "|" not in line:
                continue
            image_path, target = line.split("|", 1)
            rows.append((image_path, target))
    return rows


def main() -> None:
    if not TRAIN_LABELS.exists() or not VAL_LABELS.exists():
        raise SystemExit(f"Label files missing under {COMBINED}")

    train_rows = load_labels(TRAIN_LABELS)
    val_rows = load_labels(VAL_LABELS)
    all_rows = train_rows + val_rows

    by_text: dict[str, list[str]] = defaultdict(list)
    for image_path, target in all_rows:
        by_text[target].append(image_path)

    unique_texts = sorted(by_text.keys())
    rng = random.Random(SEED)
    rng.shuffle(unique_texts)

    val_target_count = max(1, int(len(unique_texts) * VAL_RATIO))
    val_texts = set(unique_texts[:val_target_count])
    train_texts = set(unique_texts[val_target_count:])

    new_train: list[tuple[str, str]] = []
    new_val: list[tuple[str, str]] = []
    for target, images in by_text.items():
        bucket = new_val if target in val_texts else new_train
        for image_path in images:
            bucket.append((image_path, target))

    rng.shuffle(new_train)
    rng.shuffle(new_val)

    backup_train = TRAIN_LABELS.with_suffix(".txt.leaky.bak")
    backup_val = VAL_LABELS.with_suffix(".txt.leaky.bak")
    if not backup_train.exists():
        TRAIN_LABELS.rename(backup_train)
    else:
        TRAIN_LABELS.unlink(missing_ok=True)
    if not backup_val.exists():
        VAL_LABELS.rename(backup_val)
    else:
        VAL_LABELS.unlink(missing_ok=True)

    with TRAIN_LABELS.open("w", encoding="utf-8") as handle:
        for image_path, target in new_train:
            handle.write(f"{image_path}|{target}\n")
    with VAL_LABELS.open("w", encoding="utf-8") as handle:
        for image_path, target in new_val:
            handle.write(f"{image_path}|{target}\n")

    overlap = val_texts & train_texts
    shared_samples = sum(
        1 for _, target in new_val if target in train_texts
    )

    print("==================== SPLIT REPORT ====================")
    print(f"Total samples:           {len(all_rows)}")
    print(f"Unique target strings:   {len(unique_texts)}")
    print(f"Train samples (new):     {len(new_train)}")
    print(f"Val samples   (new):     {len(new_val)}")
    print(f"Train unique texts:      {len(train_texts)}")
    print(f"Val unique texts:        {len(val_texts)}")
    print(f"Overlap (text) :         {len(overlap)}  (should be 0)")
    print(f"Val samples whose text is in train: {shared_samples}  (should be 0)")
    print()
    print(f"Backup of leaky originals:")
    print(f"  {backup_train.relative_to(ROOT)}")
    print(f"  {backup_val.relative_to(ROOT)}")
    print("======================================================")


if __name__ == "__main__":
    main()
