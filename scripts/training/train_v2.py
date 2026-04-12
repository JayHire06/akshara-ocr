import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

# PATHS:
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_LABELS = [os.path.join(ROOT_DIR, "data", "combined", "train_labels.txt")]
VAL_LABELS = os.path.join(ROOT_DIR, "data", "combined", "val_labels.txt")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "model", "checkpoints_v2")
VOCAB_FILE = os.path.join(ROOT_DIR, "vocab.json")

# Fix for real documents test path
REAL_TEST_DIR = os.path.join(ROOT_DIR, "data", "real_test", "hindi")

# STEP 1 — Build vocab
def build_vocab():
    all_chars = set()
    for labels_file in TRAIN_LABELS:
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '|' in line:
                    label = line.strip().split('|', 1)[1]
                    all_chars.update(label)

    chars = sorted(list(all_chars))
    vocab = {"<blank>": 0}
    for i, c in enumerate(chars, start=1):
        vocab[c] = i

    os.makedirs(os.path.dirname(VOCAB_FILE), exist_ok=True)
    with open(VOCAB_FILE, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Vocab size: {len(vocab)}")
    return vocab

# STEP 2 — Dataset class (on-demand disk loading — no RAM preload)
class OCRDataset(Dataset):
    def __init__(self, labels_file, vocab):
        self.samples = []
        self.vocab = vocab

        l_files = labels_file if isinstance(labels_file, list) else [labels_file]
        for lf in l_files:
            with open(lf, 'r', encoding='utf-8') as f:
                for line in f:
                    if '|' in line:
                        path, label = line.strip().split('|', 1)
                        self.samples.append((path.strip(), label.strip()))

        print(f"Dataset ready: {len(self.samples)} samples (disk mode, no RAM preload)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('L')
            w, h = img.size
            new_w = max(1, int(w * (32 / h)))
            img = img.resize((new_w, 32), Image.BILINEAR)
            img_arr = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_arr).unsqueeze(0)
        except Exception:
            img_tensor = torch.zeros(1, 32, 32)

        target = [self.vocab[c] for c in label if c in self.vocab]
        return img_tensor, torch.tensor(target, dtype=torch.long), label


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    max_w = max(img.shape[2] for img in images)
    padded_images = []
    for img in images:
        pad_w = max_w - img.shape[2]
        padded_images.append(nn.functional.pad(img, (0, pad_w)))

    images_tensor = torch.stack(padded_images)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_tensor = torch.cat(targets)

    return images_tensor, targets_tensor, target_lengths, labels


# STEP 3 — Model
class CRNN(nn.Module):
    def __init__(self, vocab_size):
        super(CRNN, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))
            )

        self.cnn = nn.Sequential(
            conv_block(1, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 512)
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, dropout=0.3, batch_first=False)
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return nn.functional.log_softmax(x, dim=2)


# STEP 4 — CTC Decoder
def ctc_decode(log_probs, vocab_inv):
    preds = torch.argmax(log_probs, dim=2)
    preds = preds.transpose(0, 1)
    decoded_strings = []
    for i in range(preds.size(0)):
        pred_seq = preds[i].tolist()
        pred_chars = []
        prev_idx = -1
        for idx in pred_seq:
            if idx != prev_idx:
                if idx != 0:
                    pred_chars.append(vocab_inv[idx])
            prev_idx = idx
        decoded_strings.append("".join(pred_chars))
    return decoded_strings


def calc_crr(preds, targets):
    correct = sum(1 for p, t in zip(preds, targets) if p == t)
    return correct / len(preds) if preds else 0.0


# STEP 5 — Training loop
def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    vocab = build_vocab()
    vocab_inv = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)

    train_dataset = OCRDataset(TRAIN_LABELS, vocab)
    val_dataset = OCRDataset(VAL_LABELS, vocab)

    # num_workers=4: parallel disk I/O while GPU computes
    # prefetch_factor=2: pre-fetch 2 batches per worker
    # persistent_workers=True: avoid worker respawn overhead each epoch
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        collate_fn=collate_fn, num_workers=4,
        pin_memory=True, prefetch_factor=2,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        collate_fn=collate_fn, num_workers=4,
        pin_memory=True, prefetch_factor=2,
        persistent_workers=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    model = CRNN(vocab_size).to(device)

    # Fine-tune from best_model_v4.pth
    v4_path = os.path.join(CHECKPOINT_DIR, "best_model_v4.pth")
    if os.path.exists(v4_path):
        pretrained_dict = torch.load(v4_path, map_location=device, weights_only=True)
        model_dict = model.state_dict()
        # Filter out mismatched layers (e.g. fc.weight when vocab size changes)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded compatible weights from best_model_v4.pth (ignored {len(model.state_dict()) - len(pretrained_dict)} mismatched layers) — fine-tuning!")
    else:
        print("WARNING: best_model_v4.pth not found. Training from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device)

    best_val_crr = -1.0
    epochs_no_improve = 0
    max_epochs = 10

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        print(f"\n--- Epoch {epoch}/{max_epochs} Started ---")

        for batch_idx, (images, targets, target_lengths, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long, device=device)

            loss = criterion(outputs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            if batch_idx % 200 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_loss /= len(train_dataset)

        model.eval()
        val_preds, val_targets = [], []
        sample_preds, sample_gts = [], []

        with torch.no_grad():
            for images, targets, target_lengths, labels in val_loader:
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                decoded = ctc_decode(outputs, vocab_inv)
                val_preds.extend(decoded)
                val_targets.extend(labels)
                if len(sample_preds) < 5:
                    for p, t in zip(decoded, labels):
                        if len(sample_preds) < 5:
                            sample_preds.append(p)
                            sample_gts.append(t)

        val_crr = calc_crr(val_preds, val_targets) * 100.0
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_crr)

        print(f"\nEpoch {epoch}/{max_epochs}")
        print(f"Train loss: {train_loss:.4f} | Val CRR: {val_crr:.2f}% | LR: {current_lr:.6f}")
        for gt, pred in zip(sample_gts, sample_preds):
            print(f"GT: {gt}  PRED: {pred}")
        print("-" * 40)

        # Test on real documents
        if os.path.exists(REAL_TEST_DIR):
            print("=== REAL DOCUMENT TEST ===")
            for fname in sorted(os.listdir(REAL_TEST_DIR))[:20]:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(REAL_TEST_DIR, fname)
                    try:
                        img = Image.open(path).convert('L')
                        w, h = img.size
                        new_w = max(1, int(w * 32 / h))
                        img = img.resize((new_w, 32), Image.BILINEAR)
                        arr = np.array(img, dtype=np.float32) / 255.0
                        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
                        with torch.no_grad():
                            out = model(tensor)
                        pred = ctc_decode(out, vocab_inv)[0]
                        print(f"  {fname}: {pred}")
                    except Exception as e:
                        print(f"  {fname}: ERROR ({e})")
            print("-" * 40)

        # Save best
        if val_crr > best_val_crr:
            best_val_crr = val_crr
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_model_v5.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"NEW BEST (v5): {val_crr:.2f}%")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 3:
                print("Early stopping triggered (patience 3).")
                break


if __name__ == '__main__':
    train()
