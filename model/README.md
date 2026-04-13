# `model/` — Architectures, Loss, Decoding, Benchmark harness

All architectures live here. Training scripts in `scripts/training/` must import these canonical classes — **never** redefine model classes inside a training script (v2 and v5 previously did, and their checkpoints became architecturally orphaned).

## Files

| File | Contents |
|---|---|
| `crnn.py` | `CRNN` (block + linear, used by v1–v5), `STN`, `MobileNetBlock`, `CRNNv6` (STN + depthwise CNN + BiLSTM, used by v6–v8) |
| `crnn_v9.py` | `CRNNv9` (STN + depthwise CNN + 6-layer Transformer encoder), `ExponentialMovingAverage` helper |
| `ctc_decoder.py` | Greedy best-path decoder shared by all architectures |
| `loss.py` | `FocalCTCLoss` (γ, α) used by v6+ training scripts |
| `benchmark_eval.py` | `_load_model`, `CheckpointLoadError`, `evaluate_checkpoint_on_labels`, `discover_default_checkpoints` |

## Architecture map

```
CRNN        = cnn.block1..5 (3x3 convs + BN + ReLU + maxpool)  → BiLSTM → linear       [v1-v5]
CRNNv6      = STN + MobileNetBlock stack (depthwise/pointwise) → BiLSTM → fc           [v6-v8]
CRNNv9      = STN + MobileNetBlock stack                       → TransformerEncoder×6  → head   [v9]
```

Key-prefix invariants used by `benchmark_eval._load_model` to auto-detect the variant:

| Detection | Architecture |
|---|---|
| `transformer.layers.*` present | `CRNNv9` |
| `depthwise` present AND no transformer keys | `CRNNv6` |
| `cnn.block*` present (and no depthwise) | `CRNN` |
| `cnn.0.0.*` + `fc.*` (legacy Sequential) | **Raises `CheckpointLoadError`** — legacy, unsupported |

## CRNNv9 spec (v9)

See [`docs/v9-design.md`](../docs/v9-design.md) for full rationale.

- Input: `(B, 1, 32, W)` grayscale
- STN: same 6-parameter affine as CRNNv6
- CNN: 5 × `MobileNetBlock` → 512-channel features, height collapsed to 1
- Projection: `Linear(512 → 256)` + sinusoidal positional encoding
- Sequence model: 6 × `nn.TransformerEncoderLayer(d_model=256, nhead=8, d_ff=1024, gelu, pre-norm)`
- Head: `LayerNorm → Linear(256 → vocab)`
- Output: `log_softmax(dim=2)` — CTC-ready `(T, B, V)`
- Parameters: ~5.87M (target ~6.5M)

## Loud loading — CheckpointLoadError

Previous eval runs silently returned random-init metrics when `load_state_dict(..., strict=False)` dropped mismatched weights. The harness now enforces:

1. A known architecture family must be detected from state-dict keys.
2. `load_state_dict(strict=False)` must retain ≥ 90% of the model's parameter tensors.
3. Sequential legacy layouts are rejected outright.

Any failure raises `CheckpointLoadError` with a precise reason. Both eval entry points catch it and print `INCOMPAT` in the final table.
