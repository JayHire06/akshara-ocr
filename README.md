<div align="center">
  <img src="./docs/assets/banner.svg" width="100%" alt="Akshara OCR Banner">
</div>

# Akshara OCR

Akshara OCR is an intelligent pipeline and application designed to recognize text accurately from complex documents. 

## Development Prerequisites

To successfully run Akshara OCR on your local machine, ensure you have the following installed:

1. **Docker / Docker Desktop**: Required for spinning up the core backend services (PostgreSQL, Redis, Celery Workers, and the Model Server).
2. **Node.js & npm**: Required to install dependencies and run the frontend interface.
3. **Git LFS**: Required explicitly to handle multi-gigabyte machine learning model binaries (`*.pth`, `*.onnx`) effectively.

## Quickstart

Start the entire stack (both frontend and backend) simultaneously with a single command:

**On Windows:**
```cmd
.\start_dev.bat
```

**On Linux / macOS / Git Bash:**
```bash
bash ./start_dev.sh
```

By default, the backend API operates on `http://localhost:8000` while the frontend is served at `http://localhost:5173`.

## Mobile Support (Native Offline Inference)

Akshara-OCR natively supports fully offline extraction locally via WASM mapping using **CapacitorJS**. You do not need the Python backend active if running the Native application!

<div align="center">
  <img src="./docs/assets/mobile_mockup.svg" width="45%" alt="Akshara Mobile Scan Screen">
  <img src="./docs/assets/mobile_mockup_history.svg" width="45%" alt="Akshara Mobile History Screen">
</div>

```bash
cd frontend
npm install
npx cap open android
```

## Training & Model Benchmarking

Akshara-OCR organizes its logic sequentially across `/scripts/training/` encompassing 5 developmental paradigm shifts (v1 through v5).

To re-train all core OCR variations from scratch and natively compare their isolated metrics, launch the benchmark orchestrator:

```bash
./scripts/run_benchmarks.sh
```

Once runs complete, execute the validation suite. This will evaluate your `best_model` binaries against explicitly unseen real-world Hindi handwritten templates to verify performance:

```bash
python scripts/inference/evaluate_all_versions.py
```