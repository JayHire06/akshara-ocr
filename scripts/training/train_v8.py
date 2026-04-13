"""
Akshara-OCR  -  train_v8.py
============================
Staged curriculum training for V8.

V8 departs from the monolithic data/combined/ pool used by v1-v7.
Instead, training progresses through independent stages, each backed
by a dedicated label file under data/v8/stages/:

  Stage 1: Base synthetic pretraining   (IIIT-Synthetic-R-Hindi)
  Stage 2: Real printed adaptation      (Mozhi-Hindi)
  Stage 3: Real handwritten adaptation  (IIIT-INDIC-HW-WORDS-Hindi)

Each stage fine-tunes the checkpoint produced by the previous stage,
with a reduced learning rate for real-data stages. The V3+ 200K cap
applies to each stage independently.

Architecture: CRNNv6 (STN + MobileNet depthwise separable CNN + BiLSTM)
Loss:         FocalCTCLoss (gamma=2.0)
Augmentation: Albumentations (via OCRDatasetV6)

Checkpoints: model/checkpoints_v8/
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
    class tqdm:
        def __init__(self, iterable=None, **kw): self._it = iterable
        def __iter__(self): return iter(self._it)
        def set_postfix(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

# Paths
ROOT_DIR       = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = ROOT_DIR / "model" / "checkpoints_v8"
STAGES_DIR     = ROOT_DIR / "data" / "v8" / "stages"
VOCAB_FILE     = ROOT_DIR / "data" / "combined" / "vocab.json"

sys.path.insert(0, str(ROOT_DIR))

from data.dataset import OCRDatasetV6
from model.crnn import CRNNv6
from model.loss import FocalCTCLoss

# ---------------------------------------------------------------------------
# Stage definitions: (directory name, description, epochs, learning rate)
# ---------------------------------------------------------------------------
STAGES = [
    {
        "name": "stage1_base_iiit_synth_r",
        "desc": "Stage 1: Base synthetic pretraining",
        "epochs": 10,
        "lr": 5e-4,
    },
    {
        "name": "stage2_printed_real_mozhi",
        "desc": "Stage 2: Real printed adaptation (Mozhi-Hindi)",
        "epochs": 10,
        "lr": 1e-4,
    },
    {
        "name": "stage3_handwritten_real_iiit_hw_words",
        "desc": "Stage 3: Real handwritten adaptation (IIIT-HW-WORDS)",
        "epochs": 10,
        "lr": 5e-5,
    },
]

# Shared hyperparameters
CFG = {
    "batch_size":   128,
    "weight_decay": 1e-4,
    "grad_clip":    5.0,
    "patience":     6,
    "num_workers":  min(8, max(1, os.cpu_count() - 2)),
    "in_channels":  1,
    "lstm_hidden":  256,
    "lstm_layers":  2,
    "lstm_dropout": 0.3,
    "train_cap":    200_000,
    "val_cap":      10_000,
}


def collate_fn(batch):
    images, targets, labels = zip(*batch)
    max_w = max(img.shape[2] for img in images)
    padded = [nn.functional.pad(img, (0, max_w - img.shape[2])) for img in images]
    img_tensor  = torch.stack(padded)
    tgt_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    tgt_cat     = torch.cat(targets)
    return img_tensor, tgt_cat, tgt_lengths, list(labels)


def ctc_greedy_decode(log_probs: torch.Tensor, vocab_inv: dict) -> list[str]:
    preds = log_probs.argmax(dim=2).transpose(0, 1)
    out = []
    for seq in preds:
        chars, prev = [], -1
        for idx in seq.tolist():
            if idx != prev and idx != 0:
                chars.append(vocab_inv.get(idx, ""))
            prev = idx
        out.append("".join(chars))
    return out


def train():
    if not VOCAB_FILE.exists():
        print(f"Vocab not found at {VOCAB_FILE}")
        sys.exit(1)

    with open(VOCAB_FILE, encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_inv = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = CHECKPOINT_DIR / f"run_v8_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[V8 Staged Curriculum Training]")
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Stages: {len(STAGES)}\n")

    model = CRNNv6(vocab_size, CFG).to(device)
    criterion = FocalCTCLoss(blank_id=0, gamma=2.0, alpha=1.0, zero_infinity=True).to(device)

    # Track best CRR across all stages
    global_best_crr = 0.0

    for stage_idx, stage in enumerate(STAGES):
        stage_name = stage["name"]
        stage_dir  = STAGES_DIR / stage_name
        train_labels = stage_dir / "train_labels.txt"
        val_labels   = stage_dir / "val_labels.txt"

        print(f"\n{'=' * 60}")
        print(f"  {stage['desc']}")
        print(f"  Labels: {stage_dir}")
        print(f"  Epochs: {stage['epochs']}  LR: {stage['lr']}")
        print(f"{'=' * 60}")

        if not train_labels.exists():
            print(f"  SKIPPED: {train_labels} not found")
            continue

        # Load datasets with V3+ 200K cap
        full_train_ds = OCRDatasetV6(str(train_labels), vocab, is_training=True)
        n_train = min(CFG["train_cap"], len(full_train_ds))
        train_ds = Subset(full_train_ds, range(n_train))

        val_ds = None
        n_val = 0
        if val_labels.exists():
            full_val_ds = OCRDatasetV6(str(val_labels), vocab, is_training=False)
            n_val = min(CFG["val_cap"], len(full_val_ds))
            val_ds = Subset(full_val_ds, range(n_val))

        print(f"  Train: {n_train} samples  Val: {n_val} samples")

        loader_kw = dict(
            collate_fn=collate_fn, num_workers=CFG["num_workers"],
            pin_memory=(device.type == "cuda"), persistent_workers=(CFG["num_workers"] > 0)
        )
        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, **loader_kw)
        val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False, **loader_kw) if val_ds else None

        # Fresh optimizer per stage (reset momentum) with stage-specific LR
        optimizer = optim.AdamW(model.parameters(), lr=stage["lr"], weight_decay=CFG["weight_decay"])
        scaler = amp.GradScaler('cuda', enabled=(device.type == "cuda"))

        total_steps = stage["epochs"] * max(len(train_loader), 1)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=stage["lr"], total_steps=total_steps,
            pct_start=0.15, anneal_strategy="cos"
        )

        stage_best_crr = 0.0

        for epoch in range(1, stage["epochs"] + 1):
            epoch_start = time.time()
            model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"S{stage_idx+1} Ep {epoch:02d} [train]", leave=False) if TQDM else train_loader
            for batch_idx, (images, targets, tgt_lengths, _) in enumerate(pbar):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                tgt_lengths = tgt_lengths.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with amp.autocast('cuda', enabled=(device.type == "cuda")):
                    outputs = model(images)
                    inp_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long, device=device)
                    loss = criterion(outputs, targets, inp_lengths, tgt_lengths)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_loss += loss.item()

                if batch_idx % 200 == 0:
                    print(f"  [S{stage_idx+1} Ep {epoch:02d}] b {batch_idx}/{len(train_loader)}  Focal Loss: {loss.item():.4f}")

            train_loss /= max(len(train_loader), 1)

            # Validation
            val_crr = 0.0
            if val_loader:
                model.eval()
                all_preds, all_gts = [], []
                with torch.no_grad():
                    for images, targets, tgt_lengths, labels in val_loader:
                        images = images.to(device, non_blocking=True)
                        with amp.autocast('cuda', enabled=(device.type == "cuda")):
                            outputs = model(images)
                        preds = ctc_greedy_decode(outputs, vocab_inv)
                        all_preds.extend(preds)
                        all_gts.extend(labels)

                val_crr = sum(p == g for p, g in zip(all_preds, all_gts)) / max(len(all_preds), 1) * 100

            elapsed = time.time() - epoch_start
            print(f"  S{stage_idx+1} Ep {epoch:02d} | Loss: {train_loss:.4f} | CRR: {val_crr:.2f}% | {elapsed:.0f}s")

            # Save best for this stage
            if val_crr > stage_best_crr:
                stage_best_crr = val_crr
                torch.save(model.state_dict(), run_dir / f"best_{stage_name}.pth")
                print(f"  [Stage checkpoint saved]")

            if val_crr > global_best_crr:
                global_best_crr = val_crr
                torch.save(model.state_dict(), run_dir / "best_v8.pth")
                print(f"  [Global best: {global_best_crr:.2f}%]")

        # Save end-of-stage checkpoint (used as init for next stage)
        torch.save(model.state_dict(), run_dir / f"end_{stage_name}.pth")
        print(f"\n  Stage complete. Best CRR: {stage_best_crr:.2f}%")

    print(f"\n{'=' * 60}")
    print(f"  V8 Curriculum Training Complete")
    print(f"  Global Best CRR: {global_best_crr:.2f}%")
    print(f"  Checkpoints: {run_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train()
