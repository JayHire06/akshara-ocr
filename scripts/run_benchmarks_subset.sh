#!/bin/bash
# Akshara-OCR Benchmark Subset Pooling Script
# Executing v7 and v8 sequentially when earlier versions are already covered.

set -uo pipefail

WORKSPACE_DIR=$(dirname $(dirname $(readlink -f $0)))
LOGS_DIR="$WORKSPACE_DIR/outputs/logs"
BENCHMARK_LOG="$LOGS_DIR/benchmark.log"

# Setup logging stream
mkdir -p "$LOGS_DIR"
echo "==========================================================" | tee -a "$BENCHMARK_LOG"
echo "      Akshara-OCR Subset Benchmark (v7-v8) Initialized" | tee -a "$BENCHMARK_LOG"
echo "==========================================================" | tee -a "$BENCHMARK_LOG"

cd "$WORKSPACE_DIR"

if [ -d "venv" ]; then
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    export PYTHONUNBUFFERED=1

    echo "Activating virtual environment..." | tee -a "$BENCHMARK_LOG"
    source venv/bin/activate
fi

FAILED_VERSIONS=()

# Run Configurations Pool
run_script_vX() {
    local script_path=$1
    local identifier=$2
    
    echo -e "\n----------------------------------------------------------" | tee -a "$BENCHMARK_LOG"
    echo "  Executing: Model Version $identifier" | tee -a "$BENCHMARK_LOG"
    echo "  Script: $script_path" | tee -a "$BENCHMARK_LOG"
    echo "----------------------------------------------------------" | tee -a "$BENCHMARK_LOG"
    
    if python -u "$script_path" 2>&1 | tee -a "$BENCHMARK_LOG"; then
        echo "----------------------------------------------------------" | tee -a "$BENCHMARK_LOG"
        echo "✔️ Completed Version $identifier" | tee -a "$BENCHMARK_LOG"
    else
        FAILED_VERSIONS+=("$identifier")
        echo "----------------------------------------------------------" | tee -a "$BENCHMARK_LOG"
        echo "❌ Failed Version $identifier" | tee -a "$BENCHMARK_LOG"
    fi
}

run_script_vX "scripts/training/train_v7.py" "v7 (V6 + Active Spelling Beam NLP Reranker Engine)"
run_script_vX "scripts/training/train_v8.py" "v8 (Staged Curriculum: Synthetic -> Printed Real -> Handwritten Real)"

echo -e "\n==========================================================" | tee -a "$BENCHMARK_LOG"
if [ ${#FAILED_VERSIONS[@]} -eq 0 ]; then
    echo "  🏁 Subset Benchmarks Complete. Logs dumped at: $BENCHMARK_LOG" | tee -a "$BENCHMARK_LOG"
else
    echo "  ⚠️ Subset benchmarks completed with failures: ${FAILED_VERSIONS[*]}" | tee -a "$BENCHMARK_LOG"
    echo "  Logs dumped at: $BENCHMARK_LOG" | tee -a "$BENCHMARK_LOG"
fi
echo "==========================================================" | tee -a "$BENCHMARK_LOG"

if [ ${#FAILED_VERSIONS[@]} -ne 0 ]; then
    exit 1
fi
