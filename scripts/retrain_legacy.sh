#!/bin/bash
# Akshara-OCR  —  Retrain legacy-architecture checkpoints (v2, v5)
# =================================================================
# Why this exists:
#   train_v2.py and train_v5.py used to define their own local CRNN
#   classes with Sequential key layout (cnn.0.0.* + fc.*). Both are
#   now fixed to import the canonical model.crnn.CRNN (cnn.block* +
#   linear.*). The existing checkpoints under model/checkpoints_v2
#   and model/checkpoints/run_v5_* were saved with the old layout
#   and are unloadable against current code. This script archives
#   them and produces fresh, architecturally-compatible checkpoints
#   against the text-disjoint train/val split built by
#   scripts/data/rebuild_text_disjoint_split.py.
#
# Safe re-runs:
#   - Old checkpoints are MOVED into model/_archive_legacy_<ts>/
#     rather than deleted, so nothing is ever lost.
#   - If retraining fails, archived checkpoints can be restored.

set -uo pipefail

WORKSPACE_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
LOGS_DIR="$WORKSPACE_DIR/outputs/logs"
LOG_FILE="$LOGS_DIR/retrain_legacy.log"
TS=$(date +%Y%m%d_%H%M%S)
ARCHIVE_DIR="$WORKSPACE_DIR/model/_archive_legacy_$TS"

mkdir -p "$LOGS_DIR" "$ARCHIVE_DIR"
cd "$WORKSPACE_DIR"

echo "==========================================================" | tee "$LOG_FILE"
echo "  Akshara-OCR  —  Legacy checkpoint retrain (v2 + v5)"      | tee -a "$LOG_FILE"
echo "  timestamp: $TS"                                            | tee -a "$LOG_FILE"
echo "  archive:   $ARCHIVE_DIR"                                   | tee -a "$LOG_FILE"
echo "==========================================================" | tee -a "$LOG_FILE"

# Activate venv if present
if [ -d "venv" ]; then
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    export PYTHONUNBUFFERED=1
    echo "Activating virtualenv..." | tee -a "$LOG_FILE"
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

# Sanity: the new text-disjoint split must exist
if [ ! -f "data/combined/train_labels.txt" ] || [ ! -f "data/combined/val_labels.txt" ]; then
    echo "❌ Missing data/combined/{train,val}_labels.txt." | tee -a "$LOG_FILE"
    echo "   Run: python scripts/data/rebuild_text_disjoint_split.py" | tee -a "$LOG_FILE"
    exit 1
fi

# Confirm the split is actually text-disjoint (paranoia check)
python - <<'PY' | tee -a "$LOG_FILE"
with open("data/combined/train_labels.txt", encoding="utf-8") as f:
    tr = {ln.split("|",1)[1].strip() for ln in f if "|" in ln}
with open("data/combined/val_labels.txt", encoding="utf-8") as f:
    va = {ln.split("|",1)[1].strip() for ln in f if "|" in ln}
overlap = tr & va
print(f"[sanity] train unique texts: {len(tr)}  val unique texts: {len(va)}  overlap: {len(overlap)}")
assert not overlap, "Split is not text-disjoint!"
PY

# Archive orphaned v2 checkpoints
if [ -d "model/checkpoints_v2" ]; then
    echo -e "\n[1/3] Archiving legacy v2 checkpoints..." | tee -a "$LOG_FILE"
    mv model/checkpoints_v2 "$ARCHIVE_DIR/checkpoints_v2" 2>/dev/null || true
fi

# Archive orphaned v5 checkpoints (legacy layout lives under model/checkpoints/run_v5_*)
echo -e "\n[2/3] Archiving legacy v5 checkpoints..." | tee -a "$LOG_FILE"
if compgen -G "model/checkpoints/run_v5_*" > /dev/null; then
    mkdir -p "$ARCHIVE_DIR/checkpoints_v5_runs"
    mv model/checkpoints/run_v5_* "$ARCHIVE_DIR/checkpoints_v5_runs/" 2>/dev/null || true
fi

# Retrain
FAILED=()
run_retrain() {
    local script=$1
    local label=$2
    echo -e "\n----------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "  Retraining $label  ($script)"                                   | tee -a "$LOG_FILE"
    echo "----------------------------------------------------------"      | tee -a "$LOG_FILE"
    if python -u "$script" 2>&1 | tee -a "$LOG_FILE"; then
        echo "✔️  $label complete"                                           | tee -a "$LOG_FILE"
    else
        echo "❌  $label FAILED"                                             | tee -a "$LOG_FILE"
        FAILED+=("$label")
    fi
}

echo -e "\n[3/3] Running fresh training..." | tee -a "$LOG_FILE"
run_retrain "scripts/training/train_v2.py" "v2 (Early Aug., canonical CRNN)"
run_retrain "scripts/training/train_v5.py" "v5 (Prod Candidate, canonical CRNN)"

echo -e "\n==========================================================" | tee -a "$LOG_FILE"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  🏁 Legacy retrain complete. Archive: $ARCHIVE_DIR"        | tee -a "$LOG_FILE"
    echo "  Next:  python scripts/inference/evaluate_all_versions.py" | tee -a "$LOG_FILE"
    exit 0
else
    echo "  ⚠️  Retrain failed for: ${FAILED[*]}"                      | tee -a "$LOG_FILE"
    echo "  Archive preserved at: $ARCHIVE_DIR"                       | tee -a "$LOG_FILE"
    exit 1
fi
