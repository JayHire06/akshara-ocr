#!/usr/bin/env bash
set -euo pipefail

SOURCE=""
LIMIT=""
DB="data/external/auto_labeled/work.db"
PROVIDER="azure"
CONCURRENCY="8"
NORMALIZED_DIR="data/external/auto_labeled/normalized"
MANIFEST="data/external/auto_labeled/manifests/manifest.json"
VOCAB="data/vocab.json"

usage() {
    echo "Usage: $0 --source {wikipedia|wikisource|gov_pdf|archive_org} --limit N [--provider azure|fake] [--concurrency N]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source) SOURCE="$2"; shift 2 ;;
        --limit)  LIMIT="$2"; shift 2 ;;
        --provider) PROVIDER="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --db) DB="$2"; shift 2 ;;
        --normalized-dir) NORMALIZED_DIR="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --vocab) VOCAB="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; usage ;;
    esac
done

[[ -n "$SOURCE" && -n "$LIMIT" ]] || usage

echo "[1/4] scrape $SOURCE --limit $LIMIT"
python scripts/data/scrape_external_pages.py --source "$SOURCE" --limit "$LIMIT" --db "$DB"

echo "[2/4] label --provider $PROVIDER --concurrency $CONCURRENCY"
python scripts/data/label_with_cloud_ocr.py --db "$DB" --provider "$PROVIDER" --concurrency "$CONCURRENCY"

echo "[3/4] filter and crop"
python scripts/data/filter_and_crop_words.py \
    --db "$DB" \
    --vocab "$VOCAB" \
    --normalized-dir "$NORMALIZED_DIR" \
    --manifest "$MANIFEST"

echo "[4/4] build stage"
python scripts/data/build_auto_labeled_stage.py \
    --train-labels "$NORMALIZED_DIR/train_labels.txt" \
    --val-labels "$NORMALIZED_DIR/val_labels.txt"

echo "pipeline complete. next: start labeling_tool and verify QA candidates."
