"""
Akshara-OCR  —  train_v7.py
════════════════════════════
The V7 NLP-Guided Architecture.

Innovations (V7 Edge + Linguistic Core):
  • Mapped Submodule: STN Spatial Transformer + MobileNet Depthwise Network
  • FocalCTCLoss: Aggressively upweights rare grammar artifacts avoiding empty blanks
  • Albumentations: Physical lens & optical simulated degradations directly inside the loader
  • NLP Beam Strategy: Native SpellChecker + Top-K Beam routing for correcting misread tensors natively.

Checkpoints explicitly mapped out to: `model/checkpoints_v7_nlp/`
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

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
CHECKPOINT_DIR = ROOT_DIR / "model" / "checkpoints_v7_nlp"
DATA_DIR       = ROOT_DIR / "data" / "combined"
TRAIN_LABELS   = DATA_DIR / "train_labels.txt"
VAL_LABELS     = DATA_DIR / "val_labels.txt"
VOCAB_FILE     = DATA_DIR / "vocab.json"

sys.path.insert(0, str(ROOT_DIR))

from data.dataset import OCRDatasetV6
from model.crnn import CRNNv6
from model.loss import FocalCTCLoss
from model.ctc_beam_search import ctc_beam_search_decoder, nlp_reranker
from nlp.spell_checker import SpellChecker

# Hyperparams
CFG = {
    "batch_size":   128,
    "max_epochs":   30,
    "max_lr":       5e-4, 
    "pct_start":    0.15, 
    "weight_decay": 1e-4,
    "grad_clip":    5.0,
    "patience":     6,
    "num_workers":  min(8, max(1, os.cpu_count() - 2)),
    "in_channels":  1,
    "lstm_hidden":  256,
    "lstm_layers":  2,
    "lstm_dropout": 0.3,
    "beam_width":   5
}

def collate_fn(batch):
    images, targets, labels = zip(*batch)
    max_w = max(img.shape[2] for img in images)
    padded = [nn.functional.pad(img, (0, max_w - img.shape[2])) for img in images]
    img_tensor  = torch.stack(padded)
    tgt_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    tgt_cat     = torch.cat(targets)
    return img_tensor, tgt_cat, tgt_lengths, list(labels)

def train():
    if not TRAIN_LABELS.exists():
        print(f"Data not found at {TRAIN_LABELS}")
        sys.exit(1)

    with open(VOCAB_FILE, encoding="utf-8") as f:
        vocab = json.load(f)
        
    vocab_inv = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run_dir = CHECKPOINT_DIR / f"run_v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[V7 NLP Configured]")
    print(f"Run directory: {run_dir}\n")

    model = CRNNv6(vocab_size, CFG).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CFG["max_lr"], weight_decay=CFG["weight_decay"])
    scaler = GradScaler(enabled=(device.type == "cuda"))
    criterion = FocalCTCLoss(blank_id=0, gamma=2.0, alpha=1.0, zero_infinity=True).to(device)

    train_ds = OCRDatasetV6(TRAIN_LABELS, vocab, is_training=True)
    val_ds   = OCRDatasetV6(VAL_LABELS, vocab, is_training=False)
    
    loader_kw = dict(
        collate_fn=collate_fn, num_workers=CFG["num_workers"],
        pin_memory=(device.type == "cuda"), persistent_workers=(CFG["num_workers"] > 0)
    )
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, **loader_kw)
    val_loader   = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False, **loader_kw)

    total_steps = CFG["max_epochs"] * max(len(train_loader), 1)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG["max_lr"], total_steps=total_steps, 
                                              pct_start=CFG["pct_start"], anneal_strategy="cos")

    # Native Linguistic Backend Initializers
    try:
        spell_checker = SpellChecker(language='hi')
        print("[NLP] Hindi NLP Trie Search Successfully Mounted.")
    except Exception as e:
        print(f"[NLP WARN] Spell Checker Failed to mount: {e}")
        spell_checker = None

    best_crr = 0.0
    for epoch in range(1, CFG["max_epochs"] + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Ep {epoch:02d} [train]", leave=False) if TQDM else train_loader
        for batch_idx, (images, targets, tgt_lengths, _) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            tgt_lengths = tgt_lengths.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
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
                print(f"  [Ep {epoch:02d}] b {batch_idx}/{len(train_loader)}  Loss (Focal V7): {loss.item():.4f}")

        train_loss /= max(len(train_loader), 1)

        # Eval Setup directly applying NLP
        model.eval()
        all_preds, all_gts = [], []
        val_loss_sum = 0.0
        
        with torch.no_grad():
            for images, targets, tgt_lengths, labels in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                tgt_lengths = tgt_lengths.to(device, non_blocking=True)
                with autocast(enabled=(device.type == "cuda")):
                    outputs = model(images)
                    inp_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long, device=device)
                    vloss = criterion(outputs, targets, inp_lengths, tgt_lengths)
                
                val_loss_sum += vloss.item()
                
                # Beam Search + NLP Linguistic Evaluation Mapping
                outputs_mapped = outputs.cpu() # (T, B, V)
                outputs_mapped = outputs_mapped.transpose(0, 1) # (B, T, V)
                
                for seq_idx in range(outputs_mapped.shape[0]):
                    seq_log_probs = outputs_mapped[seq_idx]
                    raw_candidates = ctc_beam_search_decoder(seq_log_probs, vocab_inv, beam_width=CFG['beam_width'], blank_id=0)
                    
                    if spell_checker:
                       reranked_final = nlp_reranker(raw_candidates, spell_checker, threshold_edit_distance=2)
                       all_preds.append(reranked_final)
                    else:
                       all_preds.append("".join(raw_candidates[0][0]) if raw_candidates else "")
                
                all_gts.extend(labels)

        val_crr = sum(p == g for p, g in zip(all_preds, all_gts)) / max(len(all_preds), 1) * 100
        print(f"\nEp {epoch:02d} | Train Loss: {train_loss:.4f} | Reranked CRR: {val_crr:.2f}%")
        
        for gt, pr in zip(all_gts[:3], all_preds[:3]):
            print(f"  GT: {gt} -> NLP_PR: {pr}")
            
        if val_crr > best_crr:
            best_crr = val_crr
            torch.save(model.state_dict(), run_dir / "best_v7.pth")
            print("  [★ NLP Validated Checkpoint Saved]")

if __name__ == "__main__":
    train()
