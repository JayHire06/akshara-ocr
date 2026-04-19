"""Minimal developer tool: pick a model version, upload an image, see the prediction.

Run:
    venv/bin/python -m scripts.devtool.inference_ui           # localhost:7000
    venv/bin/python -m scripts.devtool.inference_ui --port 8123

Why this exists:
    Debugging a leaderboard of 9 CRNN/Transformer variants by re-running
    benchmark_eval over 16K samples is too slow when all you want is "what
    does v2 say about THIS image vs v9?". This tool exposes the same
    checkpoint loader benchmark_eval already uses, but single-image and
    on-demand. No auth, no DB, no Prometheus — strictly a dev aid.
"""
from __future__ import annotations

import argparse
import io
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from model.benchmark_eval import (  # noqa: E402  (path tweak above)
    CheckpointLoadError,
    _load_model,
    _load_vocab_list,
    discover_default_checkpoints,
)
from model.ctc_decoder import CTCDecoder  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# {label: (checkpoint_path, loaded_model, decoder)} — lazy populated
_MODEL_CACHE: dict[str, tuple[str, torch.nn.Module, CTCDecoder]] = {}


def _refresh_checkpoint_list() -> list[tuple[str, str | None]]:
    return discover_default_checkpoints(PROJECT_ROOT)


def _get_model(label: str, ckpt_path: str) -> tuple[torch.nn.Module, CTCDecoder]:
    """Lazy-load + cache. Cache key is label so version dropdown == cache entry."""
    cached = _MODEL_CACHE.get(label)
    if cached and cached[0] == ckpt_path:
        return cached[1], cached[2]

    model, vocab_size = _load_model(ckpt_path, DEVICE)
    vocab_list = _load_vocab_list(PROJECT_ROOT, vocab_size)
    decoder = CTCDecoder(vocab=vocab_list, blank_id=0)
    _MODEL_CACHE[label] = (ckpt_path, model, decoder)
    return model, decoder


def _preprocess(pil: Image.Image) -> torch.Tensor:
    """Match benchmark_eval preprocessing exactly (grayscale, h=32, /255)."""
    pil = pil.convert("L")
    width, height = pil.size
    new_w = max(1, int(width * 32 / max(height, 1)))
    resized = pil.resize((new_w, 32))
    array = np.array(resized).astype(np.float32) / 255.0
    return torch.tensor(array).unsqueeze(0).unsqueeze(0).to(DEVICE)


def _segment_lines(
    pil: Image.Image,
    min_line_height: int = 6,
    min_gap_height: int = 4,
    smoothing_window: int = 5,
    pad: int = 2,
) -> list[tuple[int, int]]:
    """Split a multi-line image into per-line (y0, y1) boxes.

    Horizontal projection profile: dynamic-threshold binarise → per-row
    ink fraction → box-smooth to bridge character-internal holes →
    threshold vs the image's own peak density → collect runs.

    Design notes:
        * The binarisation threshold is set per-image as the midpoint of
          the 10th and 90th percentiles of the pixel intensity. Hardcoded
          0.5 fails on faint-grey Devanagari renders (min pixel value in
          the 0.25–0.35 range).
        * Narrow glyphs only fill ~10% of a row, so gap detection is
          relative: a row counts as "text" if its smoothed ink is above
          max(0.005, 0.15 × peak). Absolute thresholds miss first-line
          crops where peak density is already low.
        * Smoothing bridges the 1–2 row dead spots between character
          strokes without merging genuine line gaps (≥ min_gap_height).

    Fallback: if nothing gets detected, returns [(0, full_height)] so
    single-line usage is never worse than before.
    """
    arr = np.asarray(pil.convert("L"), dtype=np.float32) / 255.0
    h, w = arr.shape
    if h == 0 or w == 0:
        return [(0, h)]

    # Auto-invert via border sampling (global mean is unreliable on dense text)
    border_pixels = np.concatenate(
        [arr[:2, :].ravel(), arr[-2:, :].ravel(),
         arr[:, :2].ravel(), arr[:, -2:].ravel()]
    )
    border_mean = float(border_pixels.mean())
    if border_mean < 0.5:
        arr = 1.0 - arr
        border_mean = 1.0 - border_mean

    # Pick the binarisation threshold as "noticeably darker than the
    # background". The border tells us what background looks like; drop
    # a fixed 0.25 off it. This handles synthetic text on pure white
    # (bg=1.0 → th=0.75) and textured/grey backgrounds (bg=0.7 → th=0.45)
    # in one rule without getting confused by p10/p90 when ink coverage
    # is uneven across lines.
    bin_threshold = max(0.30, min(0.85, border_mean - 0.25))
    dark = arr < bin_threshold
    row_ink = dark.mean(axis=1)

    # Box-smooth so character-internal dead rows don't fragment a line
    if smoothing_window > 1 and len(row_ink) >= smoothing_window:
        kernel = np.ones(smoothing_window, dtype=np.float32) / smoothing_window
        smoothed = np.convolve(row_ink, kernel, mode="same")
    else:
        smoothed = row_ink

    # Absolute floor (catches sparse Devanagari rows) OR relative to peak
    # (catches very-dense text where absolute gets swamped by noise).
    peak = float(smoothed.max())
    if peak <= 0:
        return [(0, h)]
    text_threshold = min(0.015, max(0.005, 0.05 * peak))
    is_text = smoothed > text_threshold

    # Scan for contiguous text runs separated by real gaps
    boxes: list[tuple[int, int]] = []
    i = 0
    n = len(is_text)
    while i < n:
        if not is_text[i]:
            i += 1
            continue
        start = i
        while i < n and is_text[i]:
            i += 1
        end = i
        # Peek ahead: if the gap is small, merge with the next run
        gap_start = end
        while i < n and not is_text[i]:
            i += 1
        gap_len = i - gap_start
        if gap_len < min_gap_height and i < n:
            while i < n and is_text[i]:
                i += 1
            end = i
        if end - start >= min_line_height:
            y0 = max(0, start - pad)
            y1 = min(h, end + pad)
            boxes.append((y0, y1))

    if not boxes:
        return [(0, h)]
    return boxes


def _segment_words_in_line(
    line_pil: Image.Image,
    min_word_width: int = 3,
    min_gap_ratio: float = 0.35,
    pad: int = 1,
) -> list[tuple[int, int]]:
    """Split a single-line image into per-word (x0, x1) boxes.

    Vertical projection profile: binarise on the same border-aware rule as
    line segmentation, compute per-column ink fraction, then walk columns
    looking for contiguous low-ink gaps wider than an adaptive threshold.

    The vocab has no space token (the auto-labeled pipeline's training
    vocab dropped ' '), so without this step the model is fed the whole
    line as one "word" and emits a concatenated string. Devanagari words
    are connected by a top bar (शिरोरेखा); the bar breaks cleanly between
    words, giving a clean low-ink columnar gap to split on.

    The gap threshold is adaptive: `min_gap_ratio * median non-zero inter-
    glyph gap` within the line. Hardcoded gap sizes fail at extreme
    renders — a 48-px tall screenshot vs a 16-px thumbnail have wildly
    different between-word pixel widths.
    """
    arr = np.asarray(line_pil.convert("L"), dtype=np.float32) / 255.0
    h, w = arr.shape
    if h == 0 or w == 0:
        return [(0, w)]

    border_pixels = np.concatenate(
        [arr[:2, :].ravel(), arr[-2:, :].ravel(),
         arr[:, :2].ravel(), arr[:, -2:].ravel()]
    )
    border_mean = float(border_pixels.mean())
    if border_mean < 0.5:
        arr = 1.0 - arr
        border_mean = 1.0 - border_mean

    bin_threshold = max(0.30, min(0.85, border_mean - 0.25))
    dark = arr < bin_threshold
    col_ink = dark.mean(axis=0)

    # First pass: identify zero-ink (pure gap) columns, measure run lengths.
    # An adaptive gap threshold then = max(min_word_width, min_gap_ratio * mean).
    is_gap = col_ink < 0.02
    gap_runs: list[int] = []
    i = 0
    while i < w:
        if not is_gap[i]:
            i += 1
            continue
        start = i
        while i < w and is_gap[i]:
            i += 1
        gap_runs.append(i - start)
    if not gap_runs:
        return [(0, w)]

    # Pick the threshold at a fraction of the *largest* gap in the line.
    # For Devanagari the top bar connects a word — zero-ink columns inside
    # a word are rare and small, so the max gap length is almost always an
    # inter-word separator. Any gap within min_gap_ratio (0.35) of the
    # largest one is also treated as inter-word. Leading/trailing margin
    # gaps can dominate the max; clip to the interior by dropping the
    # first and last gap runs when there are ≥ 3 runs.
    interior = gap_runs[1:-1] if len(gap_runs) >= 3 else gap_runs
    mx = float(max(interior)) if interior else float(max(gap_runs))
    word_gap_threshold = max(min_word_width, int(min_gap_ratio * mx))

    boxes: list[tuple[int, int]] = []
    i = 0
    in_word = False
    word_start = 0
    gap_len = 0
    while i < w:
        if not is_gap[i]:
            if not in_word:
                word_start = i
                in_word = True
            gap_len = 0
            i += 1
        else:
            if in_word:
                # Look ahead to measure the gap
                gap_start = i
                while i < w and is_gap[i]:
                    i += 1
                gap_len = i - gap_start
                if gap_len >= word_gap_threshold:
                    x0 = max(0, word_start - pad)
                    x1 = min(w, gap_start + pad)
                    if x1 - x0 >= min_word_width:
                        boxes.append((x0, x1))
                    in_word = False
                # else: short gap inside a word → stay in word
            else:
                i += 1
    if in_word:
        x0 = max(0, word_start - pad)
        x1 = w
        if x1 - x0 >= min_word_width:
            boxes.append((x0, x1))

    if not boxes:
        return [(0, w)]
    return boxes


def _run_inference(label: str, ckpt_path: str, image_bytes: bytes) -> dict:
    model, decoder = _get_model(label, ckpt_path)
    pil = Image.open(io.BytesIO(image_bytes))
    full_w, full_h = pil.size

    line_boxes = _segment_lines(pil)

    line_results: list[dict] = []
    for (y0, y1) in line_boxes:
        line_crop = pil.crop((0, y0, full_w, y1))
        word_boxes = _segment_words_in_line(line_crop)

        word_texts: list[str] = []
        word_results: list[dict] = []
        for (x0, x1) in word_boxes:
            word_crop = line_crop.crop((x0, 0, x1, y1 - y0))
            tensor = _preprocess(word_crop)
            with torch.no_grad():
                log_probs = model(tensor)
                greedy = decoder.decode_best_path(log_probs)
            text = (greedy[0] if greedy else "").replace("<UNK>", "").strip()
            word_texts.append(text)
            word_results.append(
                {"bbox": [x0, y0, x1, y1], "text": text, "empty": text == ""}
            )

        line_text = " ".join(t for t in word_texts if t)
        line_results.append(
            {
                "bbox": [0, y0, full_w, y1],
                "text": line_text,
                "empty": line_text == "",
                "words": word_results,
            }
        )

    combined = "\n".join(r["text"] for r in line_results)
    return {
        "version": label,
        "checkpoint": ckpt_path,
        "prediction": combined,
        "empty": all(r["empty"] for r in line_results),
        "image_size": [full_w, full_h],
        "lines": line_results,
        "num_lines": len(line_results),
    }


# ---------------------------------------------------------------------------
# FastAPI app

app = FastAPI(title="Akshara-OCR dev inference tool", docs_url=None, redoc_url=None)


INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>akshara-ocr · dev inference</title>
  <style>
    body { font-family: ui-monospace, Menlo, Consolas, monospace;
           max-width: 780px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
    h1 { font-size: 1.15rem; letter-spacing: .02em; }
    .row { display: flex; gap: .75rem; align-items: center; margin: .5rem 0; }
    select, input[type=file], button {
      font: inherit; padding: .35rem .6rem; border: 1px solid #999; background: #fff;
    }
    button { cursor: pointer; background: #111; color: #fff; border-color: #111; }
    button:disabled { opacity: .5; cursor: wait; }
    .result { border: 1px solid #ccc; padding: 1rem; margin-top: 1rem;
              background: #fafafa; white-space: pre-wrap; }
    .pred { font-size: 1.6rem; font-family: "Noto Sans Devanagari", serif;
            padding: .5rem 0; color: #0a4; white-space: pre-line; line-height: 1.3; }
    .empty { color: #a00; }
    .lines { border-top: 1px dashed #bbb; margin-top: .6rem; padding-top: .4rem;
             font-size: .85rem; }
    .lines .ln { padding: .15rem 0; display: flex; gap: .6rem; }
    .lines .ln .idx { color: #888; min-width: 2.2rem; }
    .lines .ln .bbox { color: #888; font-size: .75rem; }
    .lines .ln .txt { font-family: "Noto Sans Devanagari", serif; color: #222; }
    .meta { font-size: .8rem; color: #666; }
    img.preview { max-height: 120px; max-width: 100%;
                  border: 1px solid #ccc; padding: .2rem; background: #fff; }
    .missing { color: #999; }
  </style>
</head>
<body>
  <h1>akshara-ocr · dev inference tool</h1>
  <p class="meta">
    Pick a checkpoint, upload an image (PNG/JPG). Preprocessing matches
    <code>model/benchmark_eval.py</code>: grayscale → height=32 → /255.
    Multi-line images are auto-split by horizontal projection before inference.
  </p>

  <form id="f">
    <div class="row">
      <label for="version">version:</label>
      <select id="version" name="version" required></select>
    </div>
    <div class="row">
      <input type="file" id="image" name="image" accept="image/*" required>
    </div>
    <div class="row">
      <button type="submit" id="go">run inference</button>
      <span class="meta" id="device"></span>
    </div>
  </form>

  <div id="out"></div>

  <script>
    const sel = document.getElementById('version');
    const out = document.getElementById('out');
    const go = document.getElementById('go');
    const dev = document.getElementById('device');

    async function loadVersions() {
      const r = await fetch('/versions').then(r => r.json());
      dev.textContent = 'device: ' + r.device;
      sel.innerHTML = '';
      for (const v of r.versions) {
        const opt = document.createElement('option');
        opt.value = v.label;
        opt.textContent = v.available
          ? v.label + '  ·  ' + v.checkpoint.split('/').slice(-2).join('/')
          : v.label + '  (no checkpoint)';
        if (!v.available) opt.disabled = true;
        sel.appendChild(opt);
      }
    }
    loadVersions();

    document.getElementById('f').addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const fd = new FormData();
      fd.append('version', sel.value);
      fd.append('image', document.getElementById('image').files[0]);

      go.disabled = true;
      out.innerHTML = '<div class="result">running…</div>';
      try {
        const r = await fetch('/infer', { method: 'POST', body: fd });
        const j = await r.json();
        if (!r.ok) {
          out.innerHTML = '<div class="result"><b>error:</b> ' +
            (j.detail || JSON.stringify(j)) + '</div>';
          return;
        }
        const url = URL.createObjectURL(document.getElementById('image').files[0]);
        let linesHtml = '';
        if (j.lines && j.lines.length > 1) {
          linesHtml = '<div class="lines">';
          j.lines.forEach((ln, i) => {
            linesHtml +=
              '<div class="ln">' +
                '<span class="idx">#' + (i + 1) + '</span>' +
                '<span class="txt' + (ln.empty ? ' empty' : '') + '">' +
                  (ln.empty ? '(empty)' : escapeHtml(ln.text)) +
                '</span>' +
                '<span class="bbox">y=' + ln.bbox[1] + '..' + ln.bbox[3] + '</span>' +
              '</div>';
          });
          linesHtml += '</div>';
        }
        out.innerHTML =
          '<div class="result">' +
          '<img class="preview" src="' + url + '">' +
          '<div class="pred' + (j.empty ? ' empty' : '') + '">' +
          (j.empty ? '(empty prediction)' : escapeHtml(j.prediction)) +
          '</div>' +
          linesHtml +
          '<div class="meta">' +
          'version: ' + j.version + '<br>' +
          'checkpoint: ' + j.checkpoint + '<br>' +
          'image size: ' + j.image_size.join('×') + '<br>' +
          'lines detected: ' + (j.num_lines || 1) + '<br>' +
          'elapsed: ' + j.elapsed_ms.toFixed(1) + ' ms' +
          '</div></div>';
      } catch (e) {
        out.innerHTML = '<div class="result"><b>error:</b> ' + e + '</div>';
      } finally {
        go.disabled = false;
      }
    });

    function escapeHtml(s) {
      return s.replace(/[&<>"']/g, c => ({
        '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
      }[c]));
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


@app.get("/versions")
async def versions():
    discovered = _refresh_checkpoint_list()
    return {
        "device": str(DEVICE),
        "versions": [
            {
                "label": label,
                "checkpoint": path or "",
                "available": path is not None,
            }
            for label, path in discovered
        ],
    }


@app.post("/infer")
async def infer(version: str = Form(...), image: UploadFile = File(...)):
    import time

    discovered = dict(_refresh_checkpoint_list())
    if version not in discovered:
        return JSONResponse({"detail": f"unknown version {version!r}"}, status_code=400)

    ckpt_path = discovered[version]
    if ckpt_path is None:
        return JSONResponse(
            {"detail": f"no checkpoint found for {version!r}"}, status_code=404
        )

    try:
        image_bytes = await image.read()
        t0 = time.perf_counter()
        result = _run_inference(version, ckpt_path, image_bytes)
        result["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
        return result
    except CheckpointLoadError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=500)
    except Exception as exc:  # surface the full traceback in dev
        return JSONResponse(
            {"detail": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()},
            status_code=500,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7000)
    args = parser.parse_args()

    import uvicorn

    print(f"[devtool] starting on http://{args.host}:{args.port}  device={DEVICE}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
