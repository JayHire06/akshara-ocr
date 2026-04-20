# Benchmark matrix — Akshara OCR

_Generated 2026-04-20 03:51 UTC, device `cuda`._

## Benchmarks

| Key | Rows | Description |
|---|---:|---|
| `combined_val` | 16,680 | Text-disjoint combined val — the canonical historical benchmark. |
| `verified_test` | 1,164 | Auto-labeled confidence-gated test split (Azure conf ≥ 0.95). |
| `auto_labeled_heldout_val` | 8,365 | Held-out half of the auto-labeled val split (never used for training or QA gating). |

## CER (Character Error Rate, %)

| Model | `combined_val` | `verified_test` | `auto_labeled_heldout_val` |
|---|---:|---:|---:|
| v1 (Base Gen.) | 96.59% | 98.73% | 99.02% |
| v2 (Early Aug.) | 3.00% | 44.73% | 43.41% |
| v3 (200K Extended) | 96.66% | 98.69% | 98.91% |
| v4 (Realistic) | 96.46% | 98.78% | 98.94% |
| v5 (Current Prod) | 4.82% | 43.72% | 40.30% |
| v6 (Edge STN) | 8.66% | 64.91% | 61.25% |
| v7 (NLP-Guided) | 8.90% | 64.58% | 61.60% |
| v8 (Staged Curriculum) | 27.99% | 16.24% | 16.90% |
| v9 (Transformer) | 8.02% | 10.90% | 2.24% |

## WER (Word Error Rate, %)

| Model | `combined_val` | `verified_test` | `auto_labeled_heldout_val` |
|---|---:|---:|---:|
| v1 (Base Gen.) | 100.00% | 100.00% | 100.00% |
| v2 (Early Aug.) | 13.12% | 91.84% | 78.92% |
| v3 (200K Extended) | 100.00% | 100.00% | 99.98% |
| v4 (Realistic) | 100.00% | 100.00% | 100.00% |
| v5 (Current Prod) | 16.68% | 88.23% | 76.45% |
| v6 (Edge STN) | 19.37% | 98.45% | 91.63% |
| v7 (NLP-Guided) | 20.05% | 98.88% | 92.78% |
| v8 (Staged Curriculum) | 81.19% | 50.26% | 40.01% |
| v9 (Transformer) | 58.03% | 31.61% | 6.79% |

## Exact-match (%)

| Model | `combined_val` | `verified_test` | `auto_labeled_heldout_val` |
|---|---:|---:|---:|
| v1 (Base Gen.) | 0.00% | 0.00% | 0.00% |
| v2 (Early Aug.) | 69.12% | 8.16% | 21.08% |
| v3 (200K Extended) | 0.00% | 0.00% | 0.02% |
| v4 (Realistic) | 0.00% | 0.00% | 0.00% |
| v5 (Current Prod) | 67.72% | 11.77% | 23.55% |
| v6 (Edge STN) | 68.90% | 1.55% | 8.37% |
| v7 (NLP-Guided) | 68.87% | 1.12% | 7.22% |
| v8 (Staged Curriculum) | 18.81% | 49.74% | 59.99% |
| v9 (Transformer) | 41.97% | 68.39% | 93.21% |

## Empty-prediction rate (%)

| Model | `combined_val` | `verified_test` | `auto_labeled_heldout_val` |
|---|---:|---:|---:|
| v1 (Base Gen.) | 0.00% | 0.00% | 0.05% |
| v2 (Early Aug.) | 0.00% | 0.00% | 0.11% |
| v3 (200K Extended) | 0.00% | 0.00% | 0.00% |
| v4 (Realistic) | 0.00% | 0.09% | 0.02% |
| v5 (Current Prod) | 0.00% | 0.00% | 0.25% |
| v6 (Edge STN) | 0.00% | 0.00% | 0.00% |
| v7 (NLP-Guided) | 0.00% | 0.00% | 0.00% |
| v8 (Staged Curriculum) | 0.22% | 0.09% | 1.58% |
| v9 (Transformer) | 0.00% | 0.00% | 0.00% |

## Architecture & load status

| Model | Checkpoint | Vocab size | Params (M) | Status |
|---|---|---:|---:|---|
| v1 (Base Gen.) | `model/checkpoints_v1/checkpoint_epoch_5.pth` | 53 | 5.32 | OK |
| v2 (Early Aug.) | `model/checkpoints_v2/best_model_v2.pth` | 52 | 5.32 | OK |
| v3 (200K Extended) | `model/checkpoints_200k/checkpoint_epoch_8.pth` | 53 | 5.32 | OK |
| v4 (Realistic) | `model/checkpoints_realistic_v2/checkpoint_epoch_9.pth` | 53 | 5.32 | OK |
| v5 (Current Prod) | `model/checkpoints/run_v5_20260413_210231/best.pth` | 52 | 5.32 | OK |
| v6 (Edge STN) | `model/checkpoints_v6_edge/run_v6_20260414_015709/best_v6.pth` | 52 | 4.17 | OK |
| v7 (NLP-Guided) | `model/checkpoints_v7_nlp/run_v7_20260414_021915/best_v7.pth` | 52 | 4.17 | OK |
| v8 (Staged Curriculum) | `model/checkpoints_v8/run_v8_20260413_154830/best_v8.pth` | 52 | 4.17 | OK |
| v9 (Transformer) | `model/checkpoints_v9/run_v9_20260420_011555/best_v9.pth` | 70 | 5.88 | OK |
