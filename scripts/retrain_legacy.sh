#!/bin/bash
# Akshara-OCR  —  Retrain pre-rebuild checkpoints against text-disjoint split
# ===========================================================================
# Why this exists:
#   After scripts/data/rebuild_text_disjoint_split.py rebuilt data/combined,
#   every checkpoint trained BEFORE the rebuild is now evaluating on images
#   it was trained on — the text-disjoint rebuild moved whole text buckets
#   between train and val, dragging individual images along with them. The
#   original "suspiciously perfect" CERs from v1/v3/v4/v6/v7 in the Week-7
#   audit were image-level memorization, not generalization.
#
#   v2 and v5 were also architecturally orphaned (local Sequential CRNN
#   class) and are already retrained under the canonical model.crnn.CRNN.
#   This script focuses on v1/v3/v4/v6/v7 so the leaderboard becomes
#   comparable end-to-end. Pass --include-v2-v5 to also redo those.
#
# Models retrained (default):
#   v1  (train_v1.py)  — plain CRNN, no augmentation, 200K cap
#   v3  (train_v3.py)  — plain CRNN, 200K cap
#   v4  (train_v4.py)  — plain CRNN, realistic augmentation
#   v6  (train_v6.py)  — STN + MobileNet + BiLSTM, FocalCTCLoss
#   v7  (train_v7.py)  — v6 + NLP-guided inference sidecar
#
# Safe re-runs:
#   - Old checkpoints are MOVED into model/_archive_prerebuild_<ts>/
#     rather than deleted, so nothing is ever lost.
#   - If retraining fails partway, archived checkpoints can be restored.

set -uo pipefail

INCLUDE_V2_V5=0
for arg in "$@"; do
    case "$arg" in
        --include-v2-v5) INCLUDE_V2_V5=1 ;;
        -h|--help)
            sed -n '1,28p' "$0"; exit 0 ;;
    esac
done

WORKSPACE_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
LOGS_DIR="$WORKSPACE_DIR/outputs/logs"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOGS_DIR/retrain_prerebuild_$TS.log"
ARCHIVE_DIR="$WORKSPACE_DIR/model/_archive_prerebuild_$TS"

mkdir -p "$LOGS_DIR" "$ARCHIVE_DIR"
cd "$WORKSPACE_DIR"

echo "==========================================================" | tee "$LOG_FILE"
echo "  Akshara-OCR  —  Pre-rebuild checkpoint retrain"            | tee -a "$LOG_FILE"
echo "  timestamp:  $TS"                                           | tee -a "$LOG_FILE"
echo "  archive:    $ARCHIVE_DIR"                                  | tee -a "$LOG_FILE"
echo "  include v2/v5: $INCLUDE_V2_V5"                             | tee -a "$LOG_FILE"
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

# Archive pre-rebuild checkpoint directories
echo -e "\n[archive] Moving pre-rebuild checkpoint trees..." | tee -a "$LOG_FILE"
archive_if_exists() {
    local src=$1
    local name
    name=$(basename "$src")
    if [ -e "$src" ]; then
        mkdir -p "$ARCHIVE_DIR"
        mv "$src" "$ARCHIVE_DIR/$name" 2>/dev/null || true
        echo "  archived $src -> $ARCHIVE_DIR/$name" | tee -a "$LOG_FILE"
    fi
}
archive_if_exists "model/checkpoints_v1"
archive_if_exists "model/checkpoints_200k"          # v3
archive_if_exists "model/checkpoints_realistic_v2"  # v4
archive_if_exists "model/checkpoints_v6_edge"
archive_if_exists "model/checkpoints_v7_nlp"
if [ "$INCLUDE_V2_V5" -eq 1 ]; then
    archive_if_exists "model/checkpoints_v2"
    if compgen -G "model/checkpoints/run_v5_*" > /dev/null; then
        mkdir -p "$ARCHIVE_DIR/checkpoints_v5_runs"
        mv model/checkpoints/run_v5_* "$ARCHIVE_DIR/checkpoints_v5_runs/" 2>/dev/null || true
        echo "  archived model/checkpoints/run_v5_* -> $ARCHIVE_DIR/checkpoints_v5_runs/" | tee -a "$LOG_FILE"
    fi
fi

# Retrain runner
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

echo -e "\n[retrain] Running fresh training on text-disjoint split..." | tee -a "$LOG_FILE"
run_retrain "scripts/training/train_v1.py" "v1 (Base Gen., canonical CRNN)"
run_retrain "scripts/training/train_v3.py" "v3 (200K Extended, canonical CRNN)"
run_retrain "scripts/training/train_v4.py" "v4 (Realistic Aug., canonical CRNN)"
run_retrain "scripts/training/train_v6.py" "v6 (Edge STN + MobileNet + BiLSTM)"
run_retrain "scripts/training/train_v7.py" "v7 (NLP-Guided)"
if [ "$INCLUDE_V2_V5" -eq 1 ]; then
    run_retrain "scripts/training/train_v2.py" "v2 (Early Aug.)"
    run_retrain "scripts/training/train_v5.py" "v5 (Prod Candidate)"
fi

echo -e "\n==========================================================" | tee -a "$LOG_FILE"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  🏁 Pre-rebuild retrain complete. Archive: $ARCHIVE_DIR"        | tee -a "$LOG_FILE"
    echo "  Next: python scripts/inference/evaluate_all_versions.py"      | tee -a "$LOG_FILE"
    exit 0
else
    echo "  ⚠️  Retrain failed for: ${FAILED[*]}"                          | tee -a "$LOG_FILE"
    echo "  Archive preserved at: $ARCHIVE_DIR"                           | tee -a "$LOG_FILE"
    exit 1
fi
