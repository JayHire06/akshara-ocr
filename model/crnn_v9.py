"""
CRNNv9 — "The Honest Leap" architecture.

Transformer-encoder replacement for the CRNNv6 BiLSTM while keeping the STN
front-end and MobileNet depthwise-separable CNN feature extractor intact.

Rationale: the BiLSTM in CRNN/CRNNv6 was the dominant capacity bottleneck for
Devanagari conjunct recognition. A compact Transformer encoder (6 layers,
d_model=256) captures long-range dependencies between glyph fragments and
trains faster per step thanks to parallel attention. See docs/v9-design.md.

Design constraints preserved from v6:
- STN stays identical so v6 data augmentation pipelines carry over.
- MobileNet CNN stays identical so ONNX export path and WebGPU inference
  footprint are unchanged.
- Output is log-softmax over (T, B, V) so the existing FocalCTCLoss and
  CTCDecoder keep working with no change.

Intentionally NOT copied from v6:
- `config['lstm_*']` keys are ignored; Transformer knobs are
  `transformer_layers`, `transformer_heads`, `transformer_dim`, etc.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.crnn import STN, MobileNetBlock


class SinusoidalPositionalEncoding(nn.Module):
    """Classic transformer positional encoding. max_len covers the longest
    feature map we expect after the CNN stride reduction."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))  # (max_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (T, B, d_model)
        t = x.size(0)
        return x + self.pe[:t]


class CRNNv9(nn.Module):
    """STN + MobileNet CNN + 6-layer Transformer encoder + linear CTC head."""

    def __init__(self, vocab_size: int, config: dict | None = None):
        super().__init__()
        cfg = config or {}

        self.in_channels = cfg.get("in_channels", 1)
        self.use_stn = cfg.get("use_stn", True)

        self.d_model = cfg.get("transformer_dim", 256)
        self.n_heads = cfg.get("transformer_heads", 8)
        self.n_layers = cfg.get("transformer_layers", 6)
        self.d_ff = cfg.get("transformer_ff", 1024)
        self.dropout = cfg.get("transformer_dropout", 0.1)

        if self.use_stn:
            self.stn = STN(in_channels=self.in_channels)

        # ── MobileNet-depthwise feature extractor (identical to CRNNv6) ──
        self.cnn = nn.Sequential(
            MobileNetBlock(self.in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MobileNetBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MobileNetBlock(128, 256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            MobileNetBlock(256, 512),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            MobileNetBlock(512, 512),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        # CNN output is (B, 512, 1, W'); we project down to d_model before the
        # Transformer so the attention cost scales with d_model not 512.
        self.feature_proj = nn.Linear(512, self.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,  # pre-norm is the stable default for CTC training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )

        self.norm_out = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H=32, W) → log_softmax: (T, B, vocab_size)"""
        if self.use_stn:
            x = self.stn(x)
        x = self.cnn(x)                     # (B, 512, 1, W')
        x = x.squeeze(2).permute(2, 0, 1)   # (W', B, 512)
        x = self.feature_proj(x)            # (W', B, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)             # (W', B, d_model)
        x = self.norm_out(x)
        x = self.head(x)                    # (W', B, vocab_size)
        return F.log_softmax(x, dim=2)


class ExponentialMovingAverage:
    """Minimal weight-EMA helper. Maintains a shadow copy of parameters for
    use at inference. See docs/v9-design.md#pillar-3 (L3)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(
                param.detach(), alpha=1.0 - self.decay
            )

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Swap the live weights for the EMA shadow. Returns the previous
        weights so the caller can restore them after evaluation."""
        backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])
