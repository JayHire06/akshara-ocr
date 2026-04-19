# Tests — Sai's module

Test suite covering the auto-labeled pipeline, training harness, and API layer.

## Layout

- `unit/` — fast, isolated tests. Includes coverage for auto-labeled pipeline components (scrapers, labeler, filters, QA export), `CTCDecoder`, and the benchmark-suite registration introduced in commit `0071d51`.
- `integration/` — multi-component tests. Includes `test_auto_labeled_pipeline_smoke.py`, an end-to-end scrape → label → filter+crop → build_stage run against the fake OCR provider.
- `benchmarks/`, `manual/` — slower / human-driven suites.
- `fixtures/` — shared test data (e.g., `fixtures/auto_labeled/page_wiki_001.png` for the pipeline smoke).
- `conftest.py` — shared pytest config.

## Running

```bash
PYTHONPATH=$(pwd) ./venv/bin/python -m pytest tests/unit tests/integration
```

The auto-labeled pipeline smoke requires `sys.executable` + `PYTHONPATH` to be inherited by subprocess — already handled inside the test (see commit `8579662`).
