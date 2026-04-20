# Akshara-OCR v9 — "The Honest Leap"

Status: **Shipped — v9 is the final build of this project for now.**
This document was written as a design spec on 2026-04-13 during the v8 post-mortem. The pillars below were executed; the "Shipped results" block immediately under this header records how the trained and deployed model actually performs. Future work items still listed in this doc (KenLM LM prior, encoder-decoder architecture, font resynthesis) are deferred to a subsequent project phase.

## Shipped results (2026-04-20)

Production checkpoint: `model/checkpoints_v9/run_v9_20260420_011555/best_v9.pth`
Exported ONNX: `frontend/public/model.onnx` (24 MB, 70-class output head)
Training corpus: `data/combined/` (233K rows after the auto-labeled merge — 83K original combined + 150K auto-labeled).

| Benchmark | CER | WER |
|---|---|---|
| Held-out auto-labeled val (8,365 rows, in-domain) | **2.24%** | **6.79%** |
| `verified_test_labels.txt` (1,164 rows, Azure conf ≥ 0.95 gate) | **10.90%** | **31.62%** |

For reference, the pre-auto v9 baseline on `verified_test_labels.txt` was CER 57.02% / WER 96.39%. The auto-labeled data merge alone drove CER to 18.41%; the bundled depth-6 + LR-scaling + auto-labeled-val retrain closed the remaining gap to 10.90%.

Architectural parameters landed on: STN + MobileNet CNN + **6-layer Transformer encoder** (`d_model=256, heads=8, d_ff=1024`, pre-norm, GELU) + linear CTC head with **70-char vocab**. Parameters: 5.88M. Training: AdamW, OneCycleLR (`max_lr=2e-4, pct_start=0.20`), Focal CTC + label smoothing (ε=0.0), EMA weights (decay=0.999), batch size 320, early-stop patience 3 on auto-labeled held-out CER.

The post-mortem-era transformer-layers cut from 6 → 3 was an overfitting mitigation appropriate for the 83K-row corpus of the time. With the 233K-row corpus now available, the design returned to 6 layers.

---

## 1. Why v9

v8's staged curriculum trained cleanly, but three unrelated bugs hid the truth:

1. **Text-leaky val split** — 79.6% of validation samples had a target string that
   also appeared verbatim in train. The "amazing" sub-1% CER numbers on v1/v3/v4
   were memorization, not generalization.
2. **Legacy CRNN architecture in v2/v5** — `train_v2.py` and `train_v5.py` defined
   their own Sequential-style `CRNN` classes whose checkpoints could not load
   into the canonical `model.crnn.CRNN`. Their metrics were pure random-init noise.
3. **Silent `strict=False` loading** — the benchmark harness used
   `load_state_dict(..., strict=False)`, so any architecture drift dropped
   parameters without warning.

All three are fixed (see `scripts/data/rebuild_text_disjoint_split.py`,
`scripts/training/train_v{2,5}.py`, `model/benchmark_eval.py`). v9's job is to
take the now-honest baseline and push CER meaningfully down.

---

## 2. Ceiling analysis

v1–v8 share a common ancestor: **CNN feature extractor → BiLSTM sequence model →
CTC loss**. Literature and internal experiments consistently show CRNN plateauing
at 5–10% CER on clean printed Devanagari. The ceiling exists because:

- BiLSTM cannot model arbitrary-range context; Devanagari conjuncts (क्ष, त्र, ज्ञ)
  span multiple timesteps and benefit from attention.
- CTC assumes conditional independence between timesteps, so the decoder throws
  away language information present in the training set.
- MobileNet-depthwise features are lightweight but visually weaker than modern
  CNN backbones for cluttered documents.

Breaking below the ceiling requires either (a) a stronger sequence model,
(b) a language prior at decode time, or (c) both. v9 does both.

---

## 3. Design pillars (catalogue)

### Pillar 1 — Architecture

| ID | Upgrade | Rationale | Effort | Expected Δ CER |
|----|---|---|---|---|
| **A1** | **BiLSTM → 6-layer Transformer encoder (d=256, h=8)** | Long-range dependencies + parallel training | Medium | −25% to −40% relative |
| A2 | Full encoder-decoder Transformer (no CTC) | Autoregressive LM baked in | Large | −40% to −60% relative |
| A3 | MobileNet → ConvNeXt-Tiny front-end | Stronger visual features | Small | −10% to −15% relative |
| A4 | Add CoordConv to STN input | Helps STN with rotated/skewed handwriting | Small | −5% to −10% relative |

### Pillar 2 — Data

| ID | Upgrade | Rationale | Effort |
|----|---|---|---|
| **D1** | Re-synthesize training set with all 50+ fonts in `fonts/` | Typeface invariance — biggest v1–v7 failure mode | Medium |
| D2 | Clean Hindi Wikipedia screenshots, auto-labelled via v8 + QA | Free real-world data | Medium |
| D3 | Aggressive augmentation (perspective, motion blur, ink bleed) | Capture-noise robustness | Small |
| **D4** | Character-frequency rebalanced sampler | Rare glyphs (ऋ ञ ढ़ ॐ) get training signal | Small |

### Pillar 3 — Loss & training

| ID | Upgrade | Rationale | Effort |
|----|---|---|---|
| **L1** | Focal CTC + label smoothing (ε = 0.1) | Prevents overconfidence | Small |
| L2 | Length-based curriculum (1 char → word → phrase) | Faster convergence for sequence models | Small |
| **L3** | EMA of weights for inference | Free +0.5–1.5% CER | Tiny |
| L4 | Mixup / CutMix at image level | Handwriting generalization | Small |

### Pillar 4 — Decoding

| ID | Upgrade | Rationale | Effort |
|----|---|---|---|
| DC1 | KenLM char-level LM beam search | CTC greedy leaves 20–40% on the table | Medium |
| **DC2** | Reuse v7's spelling-beam word reranker | Already implemented, cheap win | Tiny |
| DC3 | Temperature scaling for calibrated confidence | Useful for UI confidence display | Tiny |

---

## 4. v9 Fast Track stack (initial implementation)

We implement the **bold-marked** items first — highest ROI, fastest path,
no new heavy dependencies.

**v9.0 = v8 frame + A1 + D1 + D4 + L1 + L3 + DC2**

- **A1** — Transformer encoder replaces BiLSTM inside a new `CRNNv9` class
  (`model/crnn_v9.py`). Keeps the STN + MobileNet front-end from CRNNv6 so
  v8 training data and preprocessing pipeline carry over unchanged.
- **D1** — A new font-resynthesis script will be added
  (`scripts/data/resynthesize_font_diverse.py`, TODO in v9.1).
- **D4** — A `CharFrequencyWeightedSampler` selects rare-glyph samples more
  often during training.
- **L1** — `FocalCTCLossLS` adds label smoothing on top of the existing focal
  CTC.
- **L3** — `torch_ema.ExponentialMovingAverage` (or a hand-rolled clone)
  tracks a shadow copy of the weights; inference uses the EMA copy.
- **DC2** — Decoding reuses `nlp/spellchecker` exactly the way `train_v7.py`
  did.

Upgrades **NOT** in v9.0 (deferred to v9.1 / v9.2):

- A2, A3, A4
- D2, D3
- L2, L4
- DC1 (KenLM), DC3

Reason: each of these either needs a week of data prep (D2), a new dependency
(KenLM), or a change to the training loop (A2). Fast Track is designed to ship
in ~2 days of wall-clock compute.

---

## 5. Concrete CRNNv9 architecture

```
Input  : (B, 1, 32, W)           grayscale, 32 rows
  │
  ▼
STN (from CRNNv6)                learns 2D affine alignment
  │
  ▼
MobileNet-depthwise CNN          stride pattern makes height=1
  (cnn.0 .. cnn.4, dep-sep)      output:  (B, 512, 1, W')
  │
  ▼
squeeze + permute                (W', B, 512)
  │
  ▼
Linear(512 → 256) + posenc       (W', B, 256)   positional embedding added
  │
  ▼
TransformerEncoder(L=6, H=8,     (W', B, 256)
    d_model=256, d_ff=1024,
    dropout=0.1, batch_first=F)
  │
  ▼
LayerNorm + Linear(256 → V)      (W', B, vocab_size)
  │
  ▼
log_softmax(dim=2)               CTC-ready output
```

Parameter count target: **~6.5M** (same order as CRNNv6's ~4.2M).
Inference compute is higher than BiLSTM per forward pass, but Transformer
parallelism makes training ~1.4× faster on the same GPU in practice.

---

## 6. Training recipe

| Knob | Value | Note |
|---|---|---|
| Dataset | `data/combined/{train,val}_labels.txt` | Text-disjoint |
| Input height | 32 | Unchanged |
| Batch size | 96 | Transformers prefer larger batches; smaller than CRNNv6's 128 to fit activations |
| Epochs | 40 | CTC with Transformer typically converges by epoch 30–35 |
| Optimizer | AdamW, weight_decay=0.05 | Standard for Transformers |
| Scheduler | OneCycleLR, max_lr=3e-4, pct_start=0.10 | |
| Loss | FocalCTCLossLS(γ=2.0, α=1.0, ε=0.1) | Label smoothing |
| Grad clip | 1.0 | Tighter than CRNN (Transformers are touchier) |
| EMA decay | 0.999 | Standard value |
| Warmup | 1000 steps | Critical for Transformers |
| Mixed precision | `torch.cuda.amp` | ~1.6× speedup |

---

## 7. Success criteria

Before v9 can claim "ship":

1. **CER on text-disjoint val** ≤ 60% of v8's CER on the same split.
2. **WER ≤ 70%** of v8's WER.
3. **Rare-glyph CER** (computed on characters with <100 training occurrences)
   at least 30% lower than v8's — this is the whole point of D4.
4. No regression on v8's easy cases (clean printed hero images).
5. Checkpoint loads cleanly via `model/benchmark_eval.py` with ≥ 90% param
   match ratio.

If (1) or (2) fail, v9.0 does not ship. We go back and enable DC1 (KenLM) as
v9.1 rather than claiming a noisy improvement.

---

## 8. File inventory (what v9 adds)

```
model/crnn_v9.py                       # CRNNv9 class
scripts/training/train_v9.py           # Training loop
docs/v9-design.md                      # This doc
scripts/retrain_legacy.sh              # Fixes v2/v5 orphaned checkpoints
scripts/data/rebuild_text_disjoint_split.py   # Already shipped (Task 1)
```

Modified:

```
model/benchmark_eval.py                # v9 discovery pattern
scripts/inference/evaluate_all_versions.py    # v9 models_to_test entry
scripts/run_benchmarks.sh              # v9 training step
frontend/src/components/LandingScreen.jsx     # v9 row in BENCHMARK_ROWS
```

---

## 9. Rollback plan

v9 lives in its own files; neither `model/crnn.py` nor any v1–v8 training
script is touched. If v9 fails evaluation, delete
`model/checkpoints_v9/` and ship v8 as the production candidate. No other
rollback is required.
