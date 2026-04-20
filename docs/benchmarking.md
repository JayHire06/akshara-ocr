# Benchmark Suites

> [!IMPORTANT]
> **Current cross-version numbers** live in [`benchmark-matrix.md`](./benchmark-matrix.md) (and machine-readable form in [`benchmark-matrix.json`](./benchmark-matrix.json)). Regenerate with:
>
> ```bash
> ./venv/bin/python -m scripts.devtool.benchmark_matrix --md-out docs/benchmark-matrix.md
> ```
>
> The matrix reports CER, WER, exact-match, and empty-prediction rate for every `v1` through `v9` checkpoint across three benchmarks — `combined_val` (16,680 rows, the historical text-disjoint val), `verified_test` (1,164 rows, the auto-labeled Azure-conf-≥-0.95 split), and `auto_labeled_heldout_val` (8,365 rows, the in-domain held-out slice). It also records the checkpoint path, output-head vocab size, and parameter count per version so vocab-mismatch artefacts (v1/v3/v4 decoding under the wrong vocab) are explicit rather than silent.

> [!IMPORTANT]
> **Text-disjoint split (required).** The original `data/combined` split had ~80% text-level leakage between train and val — near-zero CER numbers on early models were memorization. The split is now rebuilt so no target string appears in both train and val. Run `python scripts/data/rebuild_text_disjoint_split.py` if the current split still contains the leaky partition (look for `*.leaky.bak` files as the tell).

> [!IMPORTANT]
> **Loud-by-default loading.** `model/benchmark_eval.py` now raises `CheckpointLoadError` on (a) legacy Sequential CRNN layout, (b) unknown architecture variants, and (c) any checkpoint loading <90% of its parameters. Evals will print `INCOMPAT` for affected versions rather than silently reporting random-init metrics. See `CheckpointLoadError` in `model/benchmark_eval.py`.

The default `data/combined/val_labels.txt` split is useful for quick iteration, but it mixes data sources and can hide where a model is actually improving. This repo also supports fixed benchmark suites so you can evaluate training recipes on separate distributions:

1. `synthetic_only`
2. `synthetic_morphed_plus_real`
3. `real_only`

If you do not have enough real photographs yet, you can carve a stable holdout from the IIIT training labels and retrain on the remainder.

## Build Suites

Example with explicit synthetic, morphed, and real label files:

```bash
python scripts/data/build_benchmark_suites.py \
  --synthetic-labels /path/to/synthetic_val_labels.txt \
  --morphed-labels /path/to/realistic_synthetic_val_labels.txt \
  --real-labels /path/to/real_photos_labels.txt \
  --output-dir data/benchmarks
```

Example with IIIT fallback when real photos are limited:

```bash
python scripts/data/build_benchmark_suites.py \
  --synthetic-labels /path/to/synthetic_val_labels.txt \
  --morphed-labels /path/to/realistic_synthetic_val_labels.txt \
  --iiit-train-labels /path/to/iiit_real/labels.txt \
  --mixed-size 1000 \
  --real-size 500 \
  --iiit-holdout-ratio 0.15
```

That command writes:

- `data/benchmarks/manifest.json`
- `data/benchmarks/synthetic_only/labels.txt`
- `data/benchmarks/synthetic_morphed_plus_real/labels.txt`
- `data/benchmarks/real_only/labels.txt`
- `data/benchmarks/iiit_holdout/holdout_labels.txt`
- `data/benchmarks/iiit_holdout/retrain_labels.txt`

Use `retrain_labels.txt` as the reduced IIIT training set and keep `holdout_labels.txt` frozen for evaluation.

## Evaluate Models

Run every saved checkpoint family against the suites:

```bash
python scripts/inference/evaluate_benchmark_suites.py \
  --benchmark-root data/benchmarks \
  --output-json tests/benchmarks/suite_report.json
```

The JSON report is grouped by benchmark, then by model version, with:

- `cer`
- `wer`
- `empty_rate`
- `samples_evaluated`

## Recommended Interpretation

- `synthetic_only`: checks whether the recognizer still reads clean rendered text.
- `synthetic_morphed_plus_real`: checks domain transfer under realistic degradation.
- `real_only`: the decision benchmark for production quality.

If `synthetic_only` improves but `real_only` is flat or worse, the model is overfitting to generated patterns rather than learning photo robustness.
