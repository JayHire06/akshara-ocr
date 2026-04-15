# Auto-Labeled External Data Pipeline

See: `docs/superpowers/specs/2026-04-16-external-auto-labeled-data-design.md`

This directory is populated by `scripts/data/run_auto_labeled_pipeline.sh`.

## Layout

- `raw/<source>/` — scraped page PNGs
- `work.db` — SQLite work queue (source of truth during a run)
- `normalized/crops/` — cropped word PNGs produced by the filter stage
- `normalized/*.txt` — pipe-format label files
- `manifests/manifest.json` — provenance, audit counts, file hashes

## Quick start

bash scripts/data/run_auto_labeled_pipeline.sh --source wikipedia --limit 5

The full run is documented in the spec's §9 integration section.
