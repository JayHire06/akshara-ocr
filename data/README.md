# `data/` — Datasets, vocab, benchmark suites

## Layout

```
data/
├── combined/                         # v1–v8 training pool, merged with auto-labeled for v9
│   ├── train/images/                 # source images
│   ├── val/images/
│   ├── train_labels.txt              # 233,901 rows after auto-labeled merge (83K pre-merge)
│   ├── train_labels.pre_auto.bak     # 83K-row snapshot taken before the auto-labeled append
│   ├── val_labels.txt                # 16,680 rows — unchanged; auto-labeled val is held separately
│   ├── vocab.json                    # char → idx, index 0 = <blank>. Now 70 entries (was 52 pre-merge).
│   ├── vocab.pre_auto.bak.json       # 52-char snapshot used by the backfill to decode old checkpoints
│   └── *.leaky.bak                   # archived pre-rebuild split (if present)
├── v8/stages/                        # v8 staged curriculum sources
│   ├── stage1_base_iiit_synth_r/
│   ├── stage2_printed_real_mozhi/
│   ├── stage3_handwritten_real_iiit_hw_words/
│   └── stage_auto_labeled_printed/   # v9 addition: pointer to normalized auto-labeled labels
├── external/auto_labeled/            # scrape → OCR → filter → verify pipeline output (see its README)
├── benchmarks/                       # named suites (synthetic_only, real_only, ...)
│   └── manifest.json
└── dataset.py                        # OCRDataset, OCRDatasetV6 (albumentations)
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

Two vocabularies exist and must stay aligned with the checkpoint's output-head size:

| Path | Size | Used by |
|---|---|---|
| `data/combined/vocab.json` (current) | **70 entries** (69 chars + `<blank>`) | v9 retrained after the auto-labeled merge |
| `data/combined/vocab.pre_auto.bak.json` | 52 entries | Pre-merge checkpoints (v1–v8). Preserved for evaluating legacy runs. |
| `./vocab.json` (repo root) | 52 entries | Frontend ONNX inference (not yet updated for the 70-char vocab). |
| `data/vocab.json` | 70 entries | Master vocab used by the auto-labeled OCR filter's OOV check. |

The jump from 52 → 70 was driven by rare chars present in the auto-labeled data that were silently dropped under the old vocab (including `ः`, `ऋ`, `ऐ`, `ऑ`, `ओ`, `औ`, `घ`, `ङ`, `झ`, `ढ`, `ॅ`, `ॉ`, `ॐ`, `ड़`, `ढ़`, and the Devanagari digits `१`, `२`, `६`, `८`). The old vocab had space ` ` as an entry; the new master vocab removes it (the model is word-level; spaces are reinserted by the inference pipeline's word segmenter, not emitted by the model).

When decoding a historical checkpoint, `scripts/devtool/backfill_trackio.py` picks the matching vocab file by output-head size. Mixing a 52-output checkpoint with the 70-char vocab produces garbage decodes.

## Benchmark suites

See [`../docs/benchmarking.md`](../docs/benchmarking.md) for the manifest format and how to build named suites (`synthetic_only`, `synthetic_morphed_plus_real`, `real_only`).
