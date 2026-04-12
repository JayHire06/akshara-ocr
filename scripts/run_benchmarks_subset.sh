#!/bin/bash
# Akshara-OCR Benchmark Subset Pooling Script
# Executing legacy iterations v3 through v5 sequentially to bypass already-trained v1/v2 models.

set -e

WORKSPACE_DIR=$(dirname $(dirname $(readlink -f $0)))
LOGS_DIR="$WORKSPACE_DIR/outputs/logs"
BENCHMARK_LOG="$LOGS_DIR/benchmark.log"

# Setup logging stream
mkdir -p "$LOGS_DIR"
echo "==========================================================" | tee -a "$BENCHMARK_LOG"
echo "      Akshara-OCR Subset Benchmark (v3-v5) Initialized" | tee -a "$BENCHMARK_LOG"
echo "==========================================================" | tee -a "$BENCHMARK_LOG"

cd "$WORKSPACE_DIR"

if [ -d "venv" ]; then
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    export PYTHONUNBUFFERED=1

    echo "Activating virtual environment..." | tee -a "$BENCHMARK_LOG"
    source venv/bin/activate
fi

# Run Configurations Pool
run_script_vX() {
    local script_path=$1
    local identifier=$2
    
    echo -e "\n----------------------------------------------------------" | tee -a "$BENCHMARK_LOG"
    echo "  Executing: Model Version $identifier" | tee -a "$BENCHMARK_LOG"
    echo "  Script: $script_path" | tee -a "$BENCHMARK_LOG"
    echo "----------------------------------------------------------" | tee -a "$BENCHMARK_LOG"
    
    python -u "$script_path" 2>&1 | tee -a "$BENCHMARK_LOG"
    
    echo "----------------------------------------------------------" | tee -a "$BENCHMARK_LOG"
    echo "✔️ Completed Version $identifier" | tee -a "$BENCHMARK_LOG"
}

run_script_vX "scripts/training/train_v3.py" "v3 (200K Extended Dictionary Pool)"
run_script_vX "scripts/training/train_v4.py" "v4 (Realistic Handcrafted Document Pool)"
run_script_vX "scripts/training/train_v5.py" "v5 (Current Best Production Candidate)"

echo -e "\n==========================================================" | tee -a "$BENCHMARK_LOG"
echo "  🏁 Subset Benchmarks Complete. Logs dumped at: $BENCHMARK_LOG" | tee -a "$BENCHMARK_LOG"
echo "==========================================================" | tee -a "$BENCHMARK_LOG"
