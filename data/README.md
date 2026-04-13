# `data/` — Datasets, vocab, benchmark suites

## Layout

```
data/
├── combined/                    # v1–v8 training pool (85K train, 15K val)
│   ├── train/images/            # source images
│   ├── val/images/
│   ├── train_labels.txt         # <abs_image_path>|<target_text>
│   ├── val_labels.txt
│   ├── vocab.json               # char → idx, index 0 = <blank>
│   └── *.leaky.bak              # archived pre-rebuild split (if present)
├── v8/stages/                   # v8 staged curriculum sources
│   ├── stage1_base_iiit_synth_r/
│   ├── stage2_printed_real_mozhi/
│   └── stage3_handwritten_real_iiit_hw_words/
├── benchmarks/                  # named suites (synthetic_only, real_only, ...)
│   └── manifest.json
└── dataset.py                   # OCRDataset, OCRDatasetV6 (albumentations)
```

## Text-disjoint split

> **Critical methodology note.**
> The original split was built by randomly partitioning images, not text. About **80% of validation target strings also appeared verbatim in the train set** — so near-zero CER numbers on early models were measuring memorization, not generalization.

To rebuild a split where no target string appears in both buckets:

```bash
python scripts/data/rebuild_text_disjoint_split.py
```

What it does:

1. Loads both label files and concatenates them.
2. Groups by unique target string (23,896 unique strings across ~100K samples).
3. Shuffles the unique-text list with a fixed seed and takes `VAL_RATIO = 0.15` of strings for val.
4. Every image whose target is in the val-text bucket goes to `val_labels.txt`; the rest go to `train_labels.txt`.
5. Old files are moved to `*.leaky.bak` so you can always restore.

The script prints a sanity report at the end; the `overlap` and `val samples whose text is in train` lines must both be **0**.

## OCRDataset variants

- **`OCRDataset`** (v1/v2/v3) — plain grayscale load + resize, no augmentation.
- **`OCRDatasetV6`** (v6+) — OpenCV load + Albumentations (shift/scale/rotate, elastic, grid distortion, gaussian noise, coarse dropout) when `is_training=True`.

All variants return `(tensor, target_indices, raw_label)` and use the shared `vocab.json` so downstream loss and decoding are identical.

## Vocab

`data/combined/vocab.json` and root `vocab.json` are identical — 52 entries, index 0 is `<blank>`. The canonical copy is `data/combined/vocab.json`; the root copy is kept for frontend ONNX inference.

## Benchmark suites

See [`../docs/benchmarking.md`](../docs/benchmarking.md) for the manifest format and how to build named suites (`synthetic_only`, `synthetic_morphed_plus_real`, `real_only`).
