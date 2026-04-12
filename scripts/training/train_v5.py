"""
Akshara-OCR  —  train_v5.py
════════════════════════════
Production training script for the v5 run.

Modern techniques:
  • Mixed Precision (AMP)      → ~2× faster on RTX 4070
  • OneCycleLR scheduler       → warmup + cosine anneal
  • Gradient clipping          → stabilises CTC training
  • Resume from checkpoint     → auto-resumes latest.pth if present
  • Organised checkpoints      → per-run directory with config + JSONL metrics
  • tqdm progress bars         → per-batch progress

Checkpoint layout:
    model/checkpoints/
        run_v5_YYYYMMDD_HHMMSS/
            config.json       ← hyperparams + git hash
            metrics.jsonl     ← one JSON per epoch
            latest.pth        ← always the last epoch (for resume)
            best.pth          ← best val CRR
            epoch_010.pth     ← periodic save every SAVE_EVERY epochs

Usage:
    python model/train_v5.py
    python model/train_v5.py --resume model/checkpoints/run_v5_xyz/latest.pth
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# ─── tqdm (graceful fall-back if not installed) ───────────────────────────────
try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, **kw):
            self._it = iterable
        def __iter__(self):
            return iter(self._it)
        def set_postfix(self, **kw):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR       = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = ROOT_DIR / "model" / "checkpoints"
DATA_DIR       = ROOT_DIR / "data" / "combined"
TRAIN_LABELS   = DATA_DIR / "train_labels.txt"
VAL_LABELS     = DATA_DIR / "val_labels.txt"
VOCAB_FILE     = DATA_DIR / "vocab.json"

# ─── Hyperparameters ─────────────────────────────────────────────────────────

CFG = {
    "batch_size":   128,
    "max_epochs":   30,
    "max_lr":       3e-4,
    "min_lr":       1e-6,
    "pct_start":    0.10,     # fraction of training spent in warmup
    "weight_decay": 1e-4,
    "grad_clip":    5.0,
    "patience":     5,        # early stopping
    "num_workers":  min(8, max(1, os.cpu_count() - 2)),
    "save_every":   5,        # save epoch_XXX.pth every N epochs
    "keep_last_n":  3,        # keep only last N periodic checkpoints
    # model architecture
    "in_channels":      1,
    "lstm_hidden":      256,
    "lstm_layers":      2,
    "lstm_dropout":     0.3,
}

# ─── Dataset ─────────────────────────────────────────────────────────────────

class OCRDataset(Dataset):
    def __init__(self, labels_file: Path, vocab: dict):
        self.vocab = vocab
        self.samples: list[tuple[str, str]] = []
        with open(labels_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "|" in line:
                    path, label = line.split("|", 1)
                    if label:
                        self.samples.append((path.strip(), label.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("L")
            w, h = img.size
            new_w = max(1, int(w * 32 / max(h, 1)))
            img = img.resize((new_w, 32), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0)   # (1, 32, W)
        except Exception:
            tensor = torch.zeros(1, 32, 64)

        encoded = [self.vocab[c] for c in label if c in self.vocab]
        return tensor, torch.tensor(encoded, dtype=torch.long), label


def collate_fn(batch):
    images, targets, labels = zip(*batch)
    max_w = max(img.shape[2] for img in images)
    padded = [nn.functional.pad(img, (0, max_w - img.shape[2])) for img in images]
    img_tensor  = torch.stack(padded)
    tgt_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    tgt_cat     = torch.cat(targets)
    return img_tensor, tgt_cat, tgt_lengths, list(labels)


# ─── Model ───────────────────────────────────────────────────────────────────

class CRNNv5(nn.Module):
    """CRNN v5 — identical architecture to v4, compatible weight loading."""

    def __init__(self, vocab_size: int, cfg: dict):
        super().__init__()

        def _block(in_c, out_c, pool):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=pool, stride=pool),
            )

        self.cnn = nn.Sequential(
            _block(cfg["in_channels"], 64,  (2, 2)),
            _block(64,  128, (2, 2)),
            _block(128, 256, (2, 1)),
            _block(256, 256, (2, 1)),
            _block(256, 512, (2, 1)),
        )

        h = cfg["lstm_hidden"]
        self.rnn = nn.LSTM(
            512, h,
            num_layers=cfg["lstm_layers"],
            bidirectional=True,
            dropout=cfg["lstm_dropout"],
            batch_first=False,
        )
        self.fc  = nn.Linear(h * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)          # (B, 512, 1, W')
        x = x.squeeze(2)         # (B, 512, W')
        x = x.permute(2, 0, 1)   # (W', B, 512)
        x, _ = self.rnn(x)       # (W', B, H*2)
        x = self.fc(x)            # (W', B, vocab)
        return nn.functional.log_softmax(x, dim=2)


# ─── Decoding / metrics ───────────────────────────────────────────────────────

def ctc_greedy_decode(log_probs: torch.Tensor, vocab_inv: dict) -> list[str]:
    preds = log_probs.argmax(dim=2).transpose(0, 1)  # (B, T)
    out = []
    for seq in preds:
        chars, prev = [], -1
        for idx in seq.tolist():
            if idx != prev and idx != 0:
                chars.append(vocab_inv.get(idx, ""))
            prev = idx
        out.append("".join(chars))
    return out


def char_recognition_rate(preds: list[str], gts: list[str]) -> float:
    if not preds:
        return 0.0
    return sum(p == g for p, g in zip(preds, gts)) / len(preds)


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(run_dir: Path, state: dict, tag: str):
    path = run_dir / f"{tag}.pth"
    torch.save(state, path)
    return path


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def write_config(run_dir: Path, cfg: dict, vocab_size: int, device: str):
    config = {
        **cfg,
        "vocab_size": vocab_size,
        "device": device,
        "git_hash": _git_hash(),
        "started_at": datetime.now().isoformat(),
        "train_labels": str(TRAIN_LABELS),
        "val_labels": str(VAL_LABELS),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return config


def append_metric(run_dir: Path, row: dict):
    with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prune_periodic_checkpoints(run_dir: Path, keep: int):
    """Keep only the last `keep` epoch_XXX.pth files."""
    ckpts = sorted(run_dir.glob("epoch_*.pth"))
    for old in ckpts[:-keep]:
        old.unlink()


# ─── Training loop ────────────────────────────────────────────────────────────

def train(resume_path: str | None = None):
    # ── Verify data ──
    if not TRAIN_LABELS.exists() or not VAL_LABELS.exists():
        print("ERROR: Training data not found. Run  python data/generate_v5_dataset.py  first.")
        sys.exit(1)

    # ── Vocab ──
    if VOCAB_FILE.exists():
        with open(VOCAB_FILE, encoding="utf-8") as f:
            vocab = json.load(f)
        print(f"Loaded vocab: {len(vocab)} characters")
    else:
        # Build vocab on the fly
        print("Building vocab from training labels...")
        chars: set[str] = set()
        for lf in [TRAIN_LABELS, VAL_LABELS]:
            with open(lf, encoding="utf-8") as f:
                for line in f:
                    if "|" in line:
                        chars.update(line.strip().split("|", 1)[1])
        vocab = {"<blank>": 0}
        for i, c in enumerate(sorted(chars), 1):
            vocab[c] = i
        VOCAB_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(VOCAB_FILE, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f"Built vocab: {len(vocab)} characters")

    vocab_inv = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)
    if device.type == "cuda":
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name}  VRAM: {gpu.total_memory // 1024 // 1024}MB")

    # ── Run directory ──
    run_name = f"run_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = CHECKPOINT_DIR / run_name
    start_epoch = 1
    best_crr = 0.0

    # ── Model ──
    model = CRNNv5(vocab_size, CFG).to(device)

    # ── Optimizer + AMP ──
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["max_lr"],
        weight_decay=CFG["weight_decay"],
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))
    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device)

    # ── Resume or fresh start ──
    if resume_path and Path(resume_path).exists():
        print(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_crr    = ckpt.get("best_crr", 0.0)
        run_dir     = Path(resume_path).parent
        run_name    = run_dir.name
        print(f"Resumed run: {run_name}  epoch={start_epoch}  best_crr={best_crr:.2f}%")
    else:
        # Try to load compatible weights from best_model_v4.pth
        v4_paths = [
            CHECKPOINT_DIR / "best_model_v4.pth",
            CHECKPOINT_DIR.parent / "best_model_v4.pth",
        ]
        for v4 in v4_paths:
            if v4.exists():
                pre = torch.load(v4, map_location=device, weights_only=True)
                md  = model.state_dict()
                compat = {k: v for k, v in pre.items() if k in md and v.shape == md[k].shape}
                md.update(compat)
                model.load_state_dict(md)
                skipped = len(md) - len(compat)
                print(f"Loaded {len(compat)} layers from best_model_v4.pth (skipped {skipped})")
                break
        else:
            print("No v4 checkpoint found — training from scratch")

        run_dir.mkdir(parents=True, exist_ok=True)
        write_config(run_dir, CFG, vocab_size, device_str)
        # Copy a README for easy orientation
        (run_dir / "README.txt").write_text(
            f"Run: {run_name}\n"
            f"metrics.jsonl — epoch metrics (tail -f to follow)\n"
            f"best.pth      — best val CRR checkpoint\n"
            f"latest.pth    — last completed epoch (for resume)\n",
            encoding="utf-8",
        )

    print(f"\nRun directory: {run_dir}\n")

    # ── Datasets ──
    train_ds = OCRDataset(TRAIN_LABELS, vocab)
    val_ds   = OCRDataset(VAL_LABELS, vocab)
    n_train, n_val = len(train_ds), len(val_ds)
    print(f"Train samples: {n_train}  Val samples: {n_val}")

    loader_kw = dict(
        collate_fn=collate_fn,
        num_workers=CFG["num_workers"],
        pin_memory=(device.type == "cuda"),
        persistent_workers=(CFG["num_workers"] > 0),
        prefetch_factor=2 if CFG["num_workers"] > 0 else None,
    )
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, **loader_kw)

    # ── Scheduler (OneCycleLR over full remaining training) ──
    total_steps = (CFG["max_epochs"] - start_epoch + 1) * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CFG["max_lr"],
        total_steps=max(total_steps, 1),
        pct_start=CFG["pct_start"],
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=1e4,
    )

    patience_counter = 0

    # ═══════════════════════════════════════════
    # Main training loop
    # ═══════════════════════════════════════════
    for epoch in range(start_epoch, CFG["max_epochs"] + 1):
        epoch_start = time.time()

        # ── Train ──
        model.train()
        train_loss = 0.0
        n_batches  = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{CFG['max_epochs']} [train]",
                    ncols=90, leave=False) if TQDM else train_loader

        for batch_idx, (images, targets, tgt_lengths, _) in enumerate(pbar):
            images     = images.to(device, non_blocking=True)
            targets    = targets.to(device, non_blocking=True)
            tgt_lengths = tgt_lengths.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                outputs = model(images)    # (T, B, vocab)
                inp_lengths = torch.full(
                    (images.size(0),), outputs.size(0),
                    dtype=torch.long, device=device
                )
                loss = criterion(outputs, targets, inp_lengths, tgt_lengths)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

            if TQDM and batch_idx % 10 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            elif batch_idx % 200 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  [{epoch:02d}] batch {batch_idx}/{n_batches}  loss={loss.item():.4f}  lr={lr:.2e}")

        train_loss /= n_batches

        # ── Validate ──
        model.eval()
        all_preds, all_gts = [], []
        val_loss_sum = 0.0

        with torch.no_grad():
            for images, targets, tgt_lengths, labels in val_loader:
                images     = images.to(device, non_blocking=True)
                targets    = targets.to(device, non_blocking=True)
                tgt_lengths = tgt_lengths.to(device, non_blocking=True)

                with autocast(enabled=(device.type == "cuda")):
                    outputs = model(images)
                    inp_lengths = torch.full(
                        (images.size(0),), outputs.size(0),
                        dtype=torch.long, device=device
                    )
                    vloss = criterion(outputs, targets, inp_lengths, tgt_lengths)
                val_loss_sum += vloss.item()

                decoded = ctc_greedy_decode(outputs, vocab_inv)
                all_preds.extend(decoded)
                all_gts.extend(labels)

        val_crr  = char_recognition_rate(all_preds, all_gts) * 100
        val_loss = val_loss_sum / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start

        # ── Print summary ──
        print(
            f"\nEpoch {epoch:02d}/{CFG['max_epochs']}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_CRR={val_crr:.2f}%  lr={current_lr:.2e}  "
            f"time={epoch_time:.0f}s"
        )
        # Sample predictions
        for gt, pred in zip(all_gts[:4], all_preds[:4]):
            print(f"  GT: {gt!r:30s}  PRED: {pred!r}")
        print("─" * 60)

        # ── Log metrics ──
        append_metric(run_dir, {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "val_crr":    round(val_crr, 4),
            "lr":         round(current_lr, 8),
            "epoch_time": round(epoch_time, 1),
            "ts":         datetime.now().isoformat(),
        })

        # ── Common checkpoint state ──
        state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler":    scaler.state_dict(),
            "val_crr":   val_crr,
            "best_crr":  best_crr,
            "cfg":       CFG,
        }

        # latest.pth — always overwrite
        save_checkpoint(run_dir, state, "latest")

        # best.pth
        if val_crr > best_crr:
            best_crr = val_crr
            state["best_crr"] = best_crr
            save_checkpoint(run_dir, state, "best")
            print(f"  ★ New best: {best_crr:.2f}%  → saved best.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        # Periodic epoch checkpoint
        if epoch % CFG["save_every"] == 0:
            save_checkpoint(run_dir, state, f"epoch_{epoch:03d}")
            prune_periodic_checkpoints(run_dir, CFG["keep_last_n"])

        # Early stopping
        if patience_counter >= CFG["patience"]:
            print(f"Early stopping — no improvement for {CFG['patience']} epochs.")
            break

    # ─── Done ─────────────────────────────────────────────────────────────────
    print(f"\n✓  Training complete.  Best val CRR: {best_crr:.2f}%")
    print(f"   Checkpoints in: {run_dir}")
    print(f"   Resume with:  python model/train_v5.py --resume {run_dir / 'latest.pth'}")

    # Export best model to ONNX
    best_ckpt = run_dir / "best.pth"
    if best_ckpt.exists():
        onnx_path = run_dir / "best_v5.onnx"
        try:
            ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
            model.eval().cpu()
            dummy = torch.zeros(1, 1, 32, 256)
            torch.onnx.export(
                model, dummy, str(onnx_path),
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {3: "width"}, "output": {0: "time"}},
                opset_version=17,
            )
            print(f"   ONNX export: {onnx_path}")
        except Exception as e:
            print(f"   ONNX export failed: {e}")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Akshara-OCR v5")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to latest.pth checkpoint to resume from",
    )
    args = parser.parse_args()
    train(resume_path=args.resume)
