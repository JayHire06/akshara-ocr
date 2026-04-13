<div align="center">
  <img src="./docs/assets/banner.svg" width="100%" alt="Akshara OCR Banner">
</div>

# Akshara OCR (V7 Standalone)

Akshara OCR is a high-performance, **native-first** Optical Character Recognition platform designed specifically for Devanagari (Hindi) script. 

> [!IMPORTANT]
> **V7 Architecture Shift**: As of version 7, Akshara OCR has migrated to a **fully standalone, decentralized architecture**. All inference, layout analysis, and history management occurs 100% on the user's device using hardware-accelerated WebGPU. The V7 engine features **Focal Loss optimization** for improved recognition of complex Hindi conjuncts.

## Key Features

1. **Hardware-Accelerated Inference**: Uses `onnxruntime-web` with WebGPU support to run V6 Hindi models at lightning speeds directly in the browser.
2. **Intelligent Line Segmentation**: Automatically detects and separates multiple lines of text in a document using Vertical Projection Profiling.
3. **100% Privacy**: No images or extracted text ever leave your device. All data is persisted locally via `localStorage`.
4. **Offline Capability**: Progressive Web App (PWA) ready and easily wrapper-compatible with CapacitorJS/Tauri.

## Quickstart

Since the application is now standalone, you only need to serve the frontend:

```bash
cd frontend
npm install
npm run dev
```

The application will be available at `http://localhost:5173`. No Docker installation or database setup is required for OCR extraction.

## Development & Research

While the **App** is standalone, the project includes raw training and evaluation assets for researchers:

### 1. Training (V1-V7)
Access the evolution of the Akshara models in `/scripts/training/`. V7 focuses on focal-loss optimization for sparse grammatical conjuncts.

### 2. Native Benchmarks
Run pure-Python evaluation metrics (CER/WER) against historical checkpoints:
```bash
python scripts/inference/evaluate_all_versions.py
```

### 3. Git LFS
**Critical**: You must have [Git LFS](https://git-lfs.github.com/) installed to pull the large `.onnx` and `.pth` binaries.

```bash
git lfs pull
```

## Mobile Support (Native Inference)

Akshara-OCR supports fully offline extraction locally via WASM mapping using **CapacitorJS**. 

<div align="center">
  <img src="./docs/assets/mobile_mockup.svg" width="45%" alt="Akshara Mobile Scan Screen">
  <img src="./docs/assets/mobile_mockup_history.svg" width="45%" alt="Akshara Mobile History Screen">
</div>

```bash
cd frontend
npx cap open android
```

---
*Built with ❤️ for the Indian Typography community.*