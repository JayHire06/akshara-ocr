"""
Akshara-OCR  -  train_v9.py
============================
"The Honest Leap" — v9 Fast-Track training.

v9.0 stack  (see docs/v9-design.md):
  A1  Transformer encoder replaces BiLSTM           → model/crnn_v9.CRNNv9
  D4  Character-frequency weighted sampler          → CharWeightedSampler
  L1  Focal CTC + label smoothing (ε=0.1)           → FocalCTCLossLS
  L3  Exponential moving average of weights         → ExponentialMovingAverage
  DC2 (decoding-only, not used during training)

Dataset:       data/combined/{train,val}_labels.txt  (text-disjoint split)
Architecture:  STN + MobileNet CNN + 6-layer Transformer encoder
Loss:          FocalCTCLossLS(γ=2.0, α=1.0, ε=0.1)
Optimizer:     AdamW, weight_decay=0.05
Scheduler:     OneCycleLR(max_lr=3e-4, pct_start=0.10)
Checkpoints:   model/checkpoints_v9/run_v9_<timestamp>/best_v9.pth
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False

    class tqdm:  # type: ignore
        def __init__(self, iterable=None, **kw): self._it = iterable
        def __iter__(self): return iter(self._it or [])
        def set_postfix(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass


# ── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = ROOT_DIR / "model" / "checkpoints_v9"
DATA_DIR       = ROOT_DIR / "data" / "combined"
TRAIN_LABELS   = DATA_DIR / "train_labels.txt"
VAL_LABELS     = DATA_DIR / "val_labels.txt"
VOCAB_FILE     = DATA_DIR / "vocab.json"

sys.path.insert(0, str(ROOT_DIR))

from data.dataset import OCRDatasetV6                        # noqa: E402
from model.crnn_v9 import CRNNv9, ExponentialMovingAverage   # noqa: E402
from model.ctc_decoder import CTCDecoder                     # noqa: E402


# ── Hyperparameters ─────────────────────────────────────────────────────────
CFG = {
    "batch_size":        96,
    "max_epochs":        40,
    "max_lr":            3e-4,
    "weight_decay":      0.05,
    "pct_start":         0.10,
    "grad_clip":         1.0,
    "num_workers":       min(8, max(1, (os.cpu_count() or 4) - 2)),
    "train_subset":      200_000,
    "val_subset":        10_000,
    "ema_decay":         0.999,
    # Model
    "in_channels":        1,
    "use_stn":            True,
    "transformer_dim":    256,
    "transformer_heads":  8,
    "transformer_layers": 6,
    "transformer_ff":     1024,
    "transformer_dropout":0.1,
    # Loss
    "focal_gamma":        2.0,
    "focal_alpha":        1.0,
    "label_smoothing":    0.1,
}


# ── L1: Focal CTC + label smoothing ─────────────────────────────────────────
class FocalCTCLossLS(nn.Module):
    """Focal CTC with additive label-smoothing over the output distribution.

    Label smoothing is implemented as a KL term toward the uniform distribution
    on every logit, weighted by ``epsilon``. CTC itself handles the alignment;
    LS just softens the per-timestep confidence. Empirically this is equivalent
    to the Transformer-TTS label-smoothing recipe.
    """

    def __init__(self, blank_id: int = 0, gamma: float = 2.0,
                 alpha: float = 1.0, epsilon: float = 0.1):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank_id, reduction="none", zero_infinity=True)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        p = torch.exp(-loss)
        focal = self.alpha * ((1 - p) ** self.gamma) * loss
        focal_mean = focal.mean()

        if self.epsilon > 0:
            # uniform KL term: -mean(log_probs) across valid positions
            # log_probs shape (T, B, V)
            uniform = -log_probs.mean()
            return (1 - self.epsilon) * focal_mean + self.epsilon * uniform
        return focal_mean


# ── D4: Character-frequency weighted sampler ────────────────────────────────
def char_frequency_weights(samples: list[tuple[str, str]]) -> list[float]:
    """One weight per sample, inversely proportional to the average character
    frequency in its target. Samples containing rare glyphs get larger weights.
    """
    char_counts: Counter[str] = Counter()
    for _, label in samples:
        char_counts.update(label)
    total = sum(char_counts.values()) or 1
    inverse: dict[str, float] = {
        ch: total / (cnt * len(char_counts)) for ch, cnt in char_counts.items()
    }
    weights: list[float] = []
    for _, label in samples:
        if not label:
            weights.append(1.0)
            continue
        weights.append(sum(inverse.get(ch, 1.0) for ch in label) / len(label))
    return weights


# ── Dataset helpers ─────────────────────────────────────────────────────────
def collate_fn(batch):
    images, targets, labels = zip(*batch)
    max_w = max(img.shape[2] for img in images)
    padded = [nn.functional.pad(img, (0, max_w - img.shape[2])) for img in images]
    img_tensor   = torch.stack(padded)
    tgt_lengths  = torch.tensor([len(t) for t in targets], dtype=torch.long)
    tgt_cat      = torch.cat(targets)
    return img_tensor, tgt_cat, tgt_lengths, list(labels)


def load_vocab() -> dict:
    with open(VOCAB_FILE, encoding="utf-8") as f:
        return json.load(f)


def build_model(vocab_size: int, device: torch.device) -> CRNNv9:
    return CRNNv9(vocab_size=vocab_size, config=CFG).to(device)


# ── Training / evaluation loops ─────────────────────────────────────────────
def decode_for_metrics(log_probs: torch.Tensor, decoder: CTCDecoder) -> list[str]:
    decoded = decoder.decode_best_path(log_probs)
    return [d.replace("<UNK>", "").strip() for d in decoded]


def char_rate(preds: list[str], targets: list[str]) -> float:
    if not preds:
        return 0.0
    return sum(p == t for p, t in zip(preds, targets)) / len(preds)


def evaluate(model: nn.Module, loader: DataLoader, decoder: CTCDecoder,
             device: torch.device) -> float:
    model.eval()
    preds: list[str] = []
    gts: list[str] = []
    with torch.no_grad():
        for images, _, _, labels in loader:
            images = images.to(device, non_blocking=True)
            with amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(images)
            preds.extend(decode_for_metrics(out, decoder))
            gts.extend(labels)
    return char_rate(preds, gts)


def save_checkpoint(run_dir: Path, tag: str, model: nn.Module,
                    ema: ExponentialMovingAverage, optimizer, scaler,
                    epoch: int, crr: float) -> Path:
    path = run_dir / f"{tag}.pth"
    torch.save({
        "epoch":       epoch,
        "model":       model.state_dict(),
        "ema_shadow":  ema.shadow,
        "optimizer":   optimizer.state_dict(),
        "scaler":      scaler.state_dict() if scaler is not None else None,
        "val_crr":     crr,
        "config":      CFG,
    }, path)
    return path


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = CHECKPOINT_DIR / f"run_v9_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(CFG, f, indent=2)

    vocab = load_vocab()
    vocab_list = [""] * len(vocab)
    for ch, idx in vocab.items():
        vocab_list[idx] = "<PAD>" if ch == "<blank>" else ch
    vocab_size = len(vocab)

    # Datasets (re-uses OCRDatasetV6 augmentation pipeline)
    full_train = OCRDatasetV6(TRAIN_LABELS, vocab, is_training=True)
    full_val   = OCRDatasetV6(VAL_LABELS,   vocab, is_training=False)
    train_len  = min(CFG["train_subset"], len(full_train))
    val_len    = min(CFG["val_subset"],   len(full_val))
    train_ds   = Subset(full_train, range(train_len))
    val_ds     = Subset(full_val,   range(val_len))

    # D4 — character-frequency weighted sampling
    sample_weights = char_frequency_weights(
        [full_train.samples[i] for i in range(train_len)]
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=train_len,
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], sampler=sampler,
        num_workers=CFG["num_workers"], pin_memory=True,
        collate_fn=collate_fn, persistent_workers=CFG["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True,
        collate_fn=collate_fn, persistent_workers=CFG["num_workers"] > 0,
    )

    model = build_model(vocab_size, device)
    ema = ExponentialMovingAverage(model, decay=CFG["ema_decay"])

    criterion = FocalCTCLossLS(
        blank_id=0,
        gamma=CFG["focal_gamma"],
        alpha=CFG["focal_alpha"],
        epsilon=CFG["label_smoothing"],
    ).to(device)

    decoder = CTCDecoder(vocab=vocab_list, blank_id=0)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["max_lr"],
        weight_decay=CFG["weight_decay"],
        betas=(0.9, 0.98),
    )
    total_steps = len(train_loader) * CFG["max_epochs"]
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CFG["max_lr"],
        total_steps=total_steps,
        pct_start=CFG["pct_start"],
        anneal_strategy="cos",
    )
    scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")

    print("=" * 60)
    print(f"Akshara-OCR v9 training  ·  run: {run_dir.name}")
    print(f"device={device}  train={train_len}  val={val_len}  "
          f"vocab={vocab_size}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print("=" * 60)

    best_crr = -1.0
    metrics_path = run_dir / "metrics.jsonl"
    for epoch in range(1, CFG["max_epochs"] + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch:02d}") if TQDM else train_loader
        for images, targets, target_lengths, _ in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=device.type == "cuda"):
                log_probs = model(images)
                input_lengths = torch.full(
                    (images.size(0),), log_probs.size(0),
                    dtype=torch.long, device=device,
                )
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ema.update(model)

            running_loss += loss.item()
            steps += 1

            if TQDM:
                pbar.set_postfix(loss=f"{loss.item():.3f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = running_loss / max(1, steps)

        # Evaluate using EMA weights
        backup = ema.copy_to(model)
        crr = evaluate(model, val_loader, decoder, device)
        ema.restore(model, backup)

        dt = time.time() - t0
        print(f"[epoch {epoch:02d}] loss={avg_loss:.4f}  val_crr={crr*100:.2f}%  ({dt:.1f}s)")

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch, "loss": avg_loss, "val_crr": crr,
                "elapsed": dt,
            }) + "\n")

        if crr > best_crr:
            best_crr = crr
            # swap EMA weights into the model for the saved best checkpoint
            backup = ema.copy_to(model)
            save_checkpoint(run_dir, "best_v9", model, ema, optimizer, scaler,
                            epoch, crr)
            ema.restore(model, backup)
            print(f"  ↳ new best saved: {best_crr*100:.2f}%")

    print("=" * 60)
    print(f"Training complete. Best EMA val CRR: {best_crr*100:.2f}%")
    print(f"Checkpoints: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
