"""
Akshara-OCR  -  train_v9.py
============================
"The Honest Leap" — v9 training (post-mortem revision).

The initial run produced 68.48% exact-match val accuracy but 8.79% CER on
the text-disjoint val — worse than the plain-CRNN v2 retrain baseline
(3.04% CER). Training loss kept dropping while val plateaued from epoch 8
onwards — textbook overfitting on short sequences. See the Week-7 entry in
PROGRESS.md for the full post-mortem. This revision addresses it:

  A1  Transformer encoder replaces BiLSTM           → CRNNv9 (depth 6 → 3)
  L1  Focal CTC (γ=2, α=1), label smoothing OFF      → FocalCTCLossLS(ε=0)
  L3  Exponential moving average of weights          → EMA(decay=0.999)
  D4  Char-frequency weighted sampling — REMOVED.   It was biasing toward
      short single-word samples (which concentrate rare glyphs) and
      starving long-sentence learning. Uniform shuffle from here on.
  Early stopping on val CER with patience=3 — the first run wasted 32
      epochs past the plateau.
  Real CER + length-bucketed metrics logged alongside exact-match so a
      flat exact-match score can't hide character-level progress again.

Dataset:       data/combined/{train,val}_labels.txt  (text-disjoint split)
Architecture:  STN + MobileNet CNN + 3-layer Transformer encoder
Loss:          FocalCTCLossLS(γ=2.0, α=1.0, ε=0.0)
Optimizer:     AdamW, weight_decay=0.05
Scheduler:     OneCycleLR(max_lr=1e-4, pct_start=0.20)
Checkpoints:   model/checkpoints_v9/run_v9_<timestamp>/best_v9.pth
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torch.utils.data import DataLoader, Subset

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

# Curriculum stages consumed by downstream staged-training orchestration.
# Purely declarative — the training loop currently trains on the combined
# labels above, but these entries document the intended stage progression so
# the auto-labeled printed stage (Task 18) is discoverable alongside synthetic
# and Mozhi stages. Paths resolve under data/v8/stages/.
CURRICULUM_STAGES: tuple[str, ...] = (
    "stage_synthetic",
    "stage_auto_labeled_printed",
    "stage_mozhi",
)

sys.path.insert(0, str(ROOT_DIR))

from data.dataset import OCRDatasetV6                           # noqa: E402
from model.benchmark_eval import levenshtein_distance, normalize_text  # noqa: E402
from model.crnn_v9 import CRNNv9, ExponentialMovingAverage      # noqa: E402
from model.ctc_decoder import CTCDecoder                        # noqa: E402
from scripts.training._trackio_logger import TrackioLogger      # noqa: E402


# ── Hyperparameters ─────────────────────────────────────────────────────────
# Post-mortem revision: depth 6 → 3, LR 3e-4 → 1e-4, pct_start 0.10 → 0.20,
# label smoothing 0.1 → 0.0, early stopping added. See module docstring.
CFG = {
    "batch_size":        320,
    "max_epochs":        40,
    "max_lr":            2e-4,    # sqrt-scaled for bs=384 (was 1e-4 at bs=96)
    "weight_decay":      0.05,
    "pct_start":         0.20,
    "grad_clip":         1.0,
    "num_workers":       min(8, max(1, (os.cpu_count() or 4) - 2)),
    "train_subset":      200_000,
    "val_subset":        10_000,
    "ema_decay":         0.999,
    "early_stop_patience": 3,
    # Model
    "in_channels":        1,
    "use_stn":            True,
    "transformer_dim":    256,
    "transformer_heads":  8,
    "transformer_layers": 6,      # restored to v9 design; was 3 post-mortem
    "transformer_ff":     1024,
    "transformer_dropout":0.1,
    # Loss
    "focal_gamma":        2.0,
    "focal_alpha":        1.0,
    "label_smoothing":    0.0,
}

# Override default val set with a held-out slice of auto-labeled data so
# early-stop tracks the domain we actually care about (real printed Hindi),
# not the synthetic+Mozhi combined/val.
VAL_LABELS_OVERRIDE = ROOT_DIR / "data/external/auto_labeled/normalized/train_val_labels.txt"


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


def _length_bucket(text: str) -> str:
    n = len(text.split())
    if n <= 1:
        return "1w"
    if n <= 4:
        return "2-4w"
    return "5+w"


def evaluate(model: nn.Module, loader: DataLoader, decoder: CTCDecoder,
             device: torch.device) -> dict:
    """Return {exact_match, cer, wer, by_length: {bucket: {n, cer, exact_match}}}.

    Exact-match is the original char_rate — full-string equality. CER is the
    character-level edit-distance rate. Length buckets let us see if the model
    is trading long-sequence accuracy for short-word memorization (the failure
    mode of the first v9 run)."""
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

    if not preds:
        return {"exact_match": 0.0, "cer": 1.0, "wer": 1.0, "by_length": {}}

    total_cer = 0.0
    total_wer = 0.0
    exact = 0
    buckets: dict[str, dict] = {"1w": {"n": 0, "cer_sum": 0.0, "exact": 0},
                                 "2-4w": {"n": 0, "cer_sum": 0.0, "exact": 0},
                                 "5+w": {"n": 0, "cer_sum": 0.0, "exact": 0}}
    for p, t in zip(preds, gts):
        p_n = normalize_text(p)
        t_n = normalize_text(t)
        if not t_n:
            cer = 1.0 if p_n else 0.0
        else:
            cer = min(1.0, levenshtein_distance(p_n, t_n) / len(t_n))
        p_words = p_n.split()
        t_words = t_n.split()
        if not t_words:
            wer = 1.0 if p_words else 0.0
        else:
            wer = min(1.0, levenshtein_distance(p_words, t_words) / len(t_words))
        is_exact = int(p_n == t_n)
        total_cer += cer
        total_wer += wer
        exact += is_exact
        b = buckets[_length_bucket(t_n)]
        b["n"] += 1
        b["cer_sum"] += cer
        b["exact"] += is_exact

    n = len(preds)
    by_length = {
        k: {
            "n": v["n"],
            "cer": (v["cer_sum"] / v["n"]) if v["n"] else None,
            "exact_match": (v["exact"] / v["n"]) if v["n"] else None,
        }
        for k, v in buckets.items()
    }
    return {
        "exact_match": exact / n,
        "cer":         total_cer / n,
        "wer":         total_wer / n,
        "by_length":   by_length,
    }


def save_checkpoint(run_dir: Path, tag: str, model: nn.Module,
                    ema: ExponentialMovingAverage, optimizer, scaler,
                    epoch: int, metrics: dict) -> Path:
    path = run_dir / f"{tag}.pth"
    torch.save({
        "epoch":       epoch,
        "model":       model.state_dict(),
        "ema_shadow":  ema.shadow,
        "optimizer":   optimizer.state_dict(),
        "scaler":      scaler.state_dict() if scaler is not None else None,
        "val_metrics": metrics,
        "config":      CFG,
    }, path)
    return path


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = CHECKPOINT_DIR / f"run_v9_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(CFG, f, indent=2)

    tracker = TrackioLogger(
        project="akshara-ocr",
        name=run_dir.name,
        config={**CFG, "arch": "CRNNv9", "dataset": "data/combined (text-disjoint)"},
    )

    vocab = load_vocab()
    vocab_list = [""] * len(vocab)
    for ch, idx in vocab.items():
        vocab_list[idx] = "<PAD>" if ch == "<blank>" else ch
    vocab_size = len(vocab)

    # Datasets (re-uses OCRDatasetV6 augmentation pipeline)
    full_train = OCRDatasetV6(TRAIN_LABELS, vocab, is_training=True)
    full_val   = OCRDatasetV6(VAL_LABELS_OVERRIDE, vocab, is_training=False)
    train_len  = min(CFG["train_subset"], len(full_train))
    val_len    = min(CFG["val_subset"],   len(full_val))
    train_ds   = Subset(full_train, range(train_len))
    val_ds     = Subset(full_val,   range(val_len))

    # Uniform shuffle. The original D4 char-frequency weighted sampler was
    # biasing toward short single-word samples (which concentrate rare
    # glyphs) and starving long-sentence learning — see post-mortem.
    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,
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

    best_cer = float("inf")
    epochs_without_improvement = 0
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
        m = evaluate(model, val_loader, decoder, device)
        ema.restore(model, backup)

        dt = time.time() - t0
        bl = m["by_length"]
        def _fmt(b: dict) -> str:
            return f"{b['cer']*100:.2f}%" if b.get("cer") is not None else "  —  "
        print(
            f"[epoch {epoch:02d}] loss={avg_loss:.4f}  "
            f"cer={m['cer']*100:.2f}%  wer={m['wer']*100:.2f}%  "
            f"exact={m['exact_match']*100:.2f}%  "
            f"|  1w={_fmt(bl['1w'])} "
            f"2-4w={_fmt(bl['2-4w'])} "
            f"5+w={_fmt(bl['5+w'])}  ({dt:.1f}s)"
        )

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch, "loss": avg_loss,
                "cer": m["cer"], "wer": m["wer"],
                "exact_match": m["exact_match"],
                "by_length": m["by_length"],
                "elapsed": dt,
            }) + "\n")

        bl_flat = {}
        for bucket_name, bucket in m["by_length"].items():
            if bucket.get("cer") is not None:
                bl_flat[f"val/cer_{bucket_name}"] = bucket["cer"]
        tracker.log(
            {
                "train/loss": avg_loss,
                "train/lr": scheduler.get_last_lr()[0],
                "val/cer": m["cer"],
                "val/wer": m["wer"],
                "val/exact_match": m["exact_match"],
                "epoch_time_sec": dt,
                **bl_flat,
            },
            step=epoch,
        )

        if m["cer"] < best_cer:
            best_cer = m["cer"]
            epochs_without_improvement = 0
            # swap EMA weights into the model for the saved best checkpoint
            backup = ema.copy_to(model)
            save_checkpoint(run_dir, "best_v9", model, ema, optimizer, scaler,
                            epoch, m)
            ema.restore(model, backup)
            print(f"  ↳ new best saved: CER {best_cer*100:.2f}%")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= CFG["early_stop_patience"]:
                print(
                    f"  ↳ early stop: {epochs_without_improvement} epochs "
                    f"without CER improvement (best={best_cer*100:.2f}%)"
                )
                break

    tracker.log({"best/val_cer": best_cer})
    tracker.finish()

    print("=" * 60)
    print(f"Training complete. Best EMA val CER: {best_cer*100:.2f}%")
    print(f"Checkpoints: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
