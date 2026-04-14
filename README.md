<div align="center">
  <img src="./docs/assets/banner.svg" width="100%" alt="Akshara OCR Banner">
</div>

# Akshara OCR

Akshara OCR is a high-performance, **native-first** Optical Character Recognition platform designed specifically for Devanagari (Hindi) script. All inference runs entirely on-device — no server round-trips, no telemetry, no uploads.

> [!IMPORTANT]
> **v8 → v9 transition in progress.** v8 (staged curriculum, CRNNv6 + MobileNet + BiLSTM) is shipping. v9 (CRNNv9 with a 6-layer Transformer encoder + EMA weights + label smoothing) is scaffolded and ready to train against the new text-disjoint split. See [docs/v9-design.md](docs/v9-design.md).

## Key Features

1. **Hardware-Accelerated Inference** — `onnxruntime-web` with WebGPU runs the exported CRNN model directly in the browser.
2. **Line Segmentation** — Vertical Projection Profiling with robust morphology for multi-line documents.
3. **100% Privacy** — No images or extracted text ever leave the device. Local history stored in `localStorage`.
4. **Offline First** — PWA-ready; Capacitor wrapper ships for Android.
5. **Honest Benchmarking** — Text-disjoint train/val split guarantees metrics measure generalization, not memorization.

## Model generations

| Version | Architecture | Notes |
|---|---|---|
| v1 | CRNN (block + linear) | Baseline from scratch |
| v2 | CRNN + early augmentation | Fine-tunes from v1 |
| v3 | CRNN, 200K extended pool | Vocab expansion |
| v4 | CRNN, realistic handcrafted documents | |
| v5 | CRNN, production candidate | OneCycleLR + AMP |
| v6 | CRNNv6 (STN + MobileNet + BiLSTM) | Edge-friendly |
| v7 | CRNNv6 + NLP spell-beam reranker | v6 inference + language prior |
| v8 | CRNNv6, staged curriculum | Synth → printed real → handwritten real |
| **v9** | **CRNNv9** (STN + MobileNet + 6-layer Transformer encoder) + EMA + label smoothing | In development — see [v9 design doc](docs/v9-design.md) |

## Quickstart — run the frontend

```bash
cd frontend
npm install
npm run dev
```

Available at `http://localhost:5173`. No backend required for OCR extraction.

## Development & research

### 1. Training scripts

All training scripts live under `scripts/training/`. Every script now imports the canonical `CRNN` or `CRNNv6` from `model/crnn.py` (v2 and v5 previously defined local classes — see **Legacy retrain** below).

```bash
# Individual run
python scripts/training/train_v9.py

# Full sweep
bash scripts/run_benchmarks.sh
```

### 2. Text-disjoint train/val split

The original combined split had ~80% text-level leakage between train and val. Rebuild the split so no target string appears in both:

```bash
python scripts/data/rebuild_text_disjoint_split.py
```

The old leaky files are preserved as `data/combined/*.leaky.bak`.

### 3. Legacy retrain (v2 + v5)

`train_v2.py` and `train_v5.py` previously defined their own `CRNN`/`CRNNv5` classes with Sequential key layout (`cnn.0.0.*` + `fc.*`) that could not load into current `model.crnn.CRNN`. Both have been fixed to import the canonical module. Any old v2/v5 checkpoints are architecturally orphaned and must be retrained:

```bash
bash scripts/retrain_legacy.sh
```

This archives the orphaned checkpoints under `model/_archive_legacy_<timestamp>/`, then retrains both on the new text-disjoint split.

### 4. Cross-version benchmarks

```bash
# Single val set (data/combined/val)
python scripts/inference/evaluate_all_versions.py

# Named benchmark suites with manifest
python scripts/inference/evaluate_benchmark_suites.py
```

The evaluator is now **loud-by-default**: any checkpoint whose state dict loads <90% of its parameters raises `CheckpointLoadError` and prints `INCOMPAT` in the report instead of silently reporting random-init metrics.

### 5. v9 development (Transformer encoder)

`docs/v9-design.md` is the authoritative spec. Short version:

- `model/crnn_v9.py` — `CRNNv9` class (STN + MobileNet CNN + 6-layer Transformer encoder + CTC head) and `ExponentialMovingAverage` helper.
- `scripts/training/train_v9.py` — training loop with AdamW, OneCycleLR, focal CTC + label smoothing, EMA weights, character-frequency weighted sampler.

To train v9 against the text-disjoint split:

```bash
python scripts/training/train_v9.py
```

Checkpoints land in `model/checkpoints_v9/run_v9_<timestamp>/best_v9.pth`.

### 6. Git LFS

**Critical**: install [Git LFS](https://git-lfs.github.com/) to pull the `.onnx` / `.pth` binaries.

```bash
git lfs pull
```

## Mobile support

Fully-offline extraction via Capacitor.

<div align="center">
  <img src="./docs/assets/mobile_mockup.svg" width="45%" alt="Akshara Mobile Scan Screen">
  <img src="./docs/assets/mobile_mockup_history.svg" width="45%" alt="Akshara Mobile History Screen">
</div>

```bash
cd frontend
npx cap open android
```
