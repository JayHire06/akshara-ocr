# Auto-Labeled External Data Pipeline

See: `docs/superpowers/specs/2026-04-16-external-auto-labeled-data-design.md`

This directory is populated by `scripts/data/run_auto_labeled_pipeline.sh`.

## Layout

- `raw/<source>/` — scraped page PNGs. `<source>` is one of `wikipedia`, `wikisource`, `gov_pdf`, `archive_org`.
- `work.db` — SQLite work queue (source of truth during a run). Holds `pages` and `words` with status transitions.
- `normalized/crops/` — cropped word PNGs (32-px height) produced by the filter stage.
- `normalized/train_labels.txt` — pipe-format `<crop_path>|<text>` lines.
- `normalized/val_labels.txt` — 10% stratified-by-source auto-val split.
- `normalized/train_val_labels.txt` / `normalized/held_out_val_labels.txt` — 50/50 split of `val_labels.txt` added during the v9 retrain: the `train_val` half becomes the early-stop signal for training, the `held_out` half is reserved as an extra benchmark.
- `normalized/qa_candidates.txt` — 500-per-source stratified set with trailing `|<wid>` for round-tripping verdicts.
- `normalized/verified_test_labels.txt` — human-verifiable test split. SHA256-pinned in the manifest and written atomically via `.tmp + os.replace`.
- `manifests/manifest.json` — provenance, audit counts, file hashes. `verified_test_labels.{path,sha256,count}` records the frozen test set.

## Production run (2026-04-19)

First real run landed the following, across three sources (`gov_pdf` skipped because its seed list was a placeholder, see the `feat(auto-labeled): scraper reliability` commit):

| | wikipedia | wikisource | archive_org | Total |
|---|---|---|---|---|
| Pages scraped | 255 | 220 | 170 | 645 |
| Blank-render rejects | 106 | 35 | 20 | 161 |
| Pages labeled by Azure | 146 | 185 | 150 | 481 |
| Raw word boxes | 97,996 | 153,752 | 27,550 | 279,298 |

Filter pass yielded:
- Kept train rows: **150,581**
- Auto-val rows: 16,730
- QA candidates: 1,500 (500/source × 3 sources)
- All four disjointness audits: **0**

From the 1,500 QA candidates we selected **1,164** rows for the verified test split; the remaining 336 were excluded.
- `verified_test_labels.txt`: **1,164 rows**. SHA256 `efbd289419dd3122a602d238b20c7da0cb043984de79c200ca0054f8f542b493`.

Azure Document Intelligence cost: ~$0.72 total. The pipeline smoke at `--source wikipedia --limit 5 --provider azure` costs ~$0.01.

## Quick start

```bash
bash scripts/data/run_auto_labeled_pipeline.sh --source wikipedia --limit 5
```

Requires `AZURE_DOC_INTEL_ENDPOINT` and `AZURE_DOC_INTEL_KEY` in env. Run the smoke first — spec §7.2.2 mandates it before any full-scale ingest. The full run is documented in the spec's §9 integration section.

## QA review workflow

```bash
PYTHONPATH=$(pwd) ./venv/bin/uvicorn data.labeling_tool.main:app --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/qa` — the reviewer UI round-robins across source buckets. Keyboard: Enter/Space = accept, `e` = edit, `r` = reject, `b` = back, Esc = cancel edit. The `/qa/export` endpoint atomically writes `verified_test_labels.txt` and updates the manifest's SHA256 block.
