# Preprocessing — Jay's module

Page-level image preprocessing run before OCR: binarisation, denoise, deskew, text-line segmentation, and Devanagari-specific shirorekha handling.

## Modules

- `binarize.py` — threshold selection for mixed-background scans.
- `denoise.py` — noise filtering.
- `deskew.py` — rotation correction via Radon / Hough-based angle detection.
- `normalize.py` — size/contrast normalisation before downstream stages.
- `segment.py` — text-line detection; separate from the line segmenter in the frontend (`frontend/src/...`) and the dev inference UI's `_segment_lines`.
- `shirorekha.py` — Devanagari top-bar detection/removal. Same glyph feature that makes word segmentation in the auto-labeled pipeline's inference path reliable (see `scripts/devtool/inference_ui.py::_segment_words_in_line`).
- `pipeline.py` — composes the above into a single preprocess call.

## Relationship to the model

Outputs feed either the server-side OCR router (`api/routers/ocr.py`) or the offline v9 retrain's labels pipeline. The model itself is word-level (no space token); line/word segmentation happens here and in the dev inference UI, not inside the model.
