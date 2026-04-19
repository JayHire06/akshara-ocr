# Akshara-OCR — Documentation Index

## Model & training

- **[v9 Design — "The Honest Leap"](./v9-design.md)** — Transformer encoder, EMA, label smoothing, text-disjoint methodology, Fast-Track stack. v9 is now trained and in production; see the root README for current CER/WER numbers.
- [v8 Data Preparation Plan](./v8-data-preparation-plan.md) — Staged curriculum sources and size targets.
- [External Auto-Labeled Data Design](./superpowers/specs/2026-04-16-external-auto-labeled-data-design.md) — Scrape → cloud-OCR → filter → human-verified QA pipeline that drives the 150K-row auto-labeled training stage and the `verified_test_labels.txt` benchmark. See also [data/external/auto_labeled/README.md](../data/external/auto_labeled/README.md) for the production run's numbers.

## Benchmarks & evaluation

- [Benchmark Suites](./benchmarking.md) — Named suites, manifest format, and running the suite evaluator.
- `auto_labeled_verified_printed` — new fourth benchmark alongside `synthetic_only`, `synthetic_morphed_plus_real`, and `real_only`. Per-source (wikipedia / wikisource / archive_org) breakdown available. Registered via `KNOWN_BENCHMARK_SUITES` in `model/benchmark_eval.py`.
- `scripts/devtool/backfill_trackio.py` — publishes historical v1–v8 checkpoint metrics to the Trackio dashboard alongside live v9 runs.

## Frontend

- [Frontend Review](./frontend-review.md)
- [Design Handoff](./design-handoff.md)
- [Demo Storyboard](./demo-storyboard.md)

## Operations

- [Security Audit](./security-audit.md)
