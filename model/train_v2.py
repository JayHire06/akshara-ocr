import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

# PATHS:
TRAIN_LABELS = r"c:\Users\jayhi\Downloads\akshara-ocr\data\combined\train_labels.txt"
VAL_LABELS = r"c:\Users\jayhi\Downloads\akshara-ocr\data\combined\val_labels.txt"
CHECKPOINT_DIR = r"c:\Users\jayhi\Downloads\akshara-ocr\model\checkpoints"
VOCAB_FILE = r"c:\Users\jayhi\Downloads\akshara-ocr\data\vocab.json"

# STEP 1 — Build vocab
def build_vocab():
    all_chars = set()
    with open(TRAIN_LABELS, 'r', encoding='utf-8') as f:
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

# STEP 2 — Dataset class
class OCRDataset(Dataset):
    def __init__(self, labels_file, vocab):
        self.samples = []
        self.images = []  # preloaded images
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '|' in line:
                    path, label = line.strip().split('|', 1)
                    self.samples.append((path.strip(), label.strip()))
        
        print(f"Preloading {len(self.samples)} images into RAM...")
        for i, (path, label) in enumerate(self.samples):
            try:
                img = Image.open(path).convert('L')
                w, h = img.size
                new_w = max(1, int(w * (32 / h)))
                img = img.resize((new_w, 32), Image.BILINEAR)
                img_arr = np.array(img, dtype=np.float32) / 255.0
                self.images.append(torch.from_numpy(img_arr).unsqueeze(0))
            except:
                self.images.append(None)
            if (i+1) % 10000 == 0:
                print(f"  Loaded {i+1}/{len(self.samples)}")
        print("Preload complete.")
        self.vocab = vocab
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_tensor = self.images[idx]
        label = self.samples[idx][1]
        if img_tensor is None:
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
        x = self.cnn(x) # (batch, 512, 1, W)
        x = x.squeeze(2) # (batch, 512, W)
        x = x.permute(2, 0, 1) # (W, batch, 512)
        
        x, _ = self.rnn(x) # (W, batch, 512)
        x = self.fc(x) # (W, batch, vocab_size)
        return nn.functional.log_softmax(x, dim=2)

# STEP 7 — CTC Decoder
def ctc_decode(log_probs, vocab_inv):
    # log_probs: (W, batch, vocab_size)
    preds = torch.argmax(log_probs, dim=2) # (W, batch)
    preds = preds.transpose(0, 1) # (batch, W)
    
    decoded_strings = []
    for i in range(preds.size(0)):
        pred_seq = preds[i].tolist()
        pred_chars = []
        prev_idx = -1
        for idx in pred_seq:
            if idx != prev_idx:
                if idx != 0: # 0 is blank
                    pred_chars.append(vocab_inv[idx])
            prev_idx = idx
        decoded_strings.append("".join(pred_chars))
    return decoded_strings

def calc_crr(preds, targets):
    correct = sum(1 for p, t in zip(preds, targets) if p == t)
    return correct / len(preds) if preds else 0.0

# STEP 4 — Training loop
def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    vocab = build_vocab()
    vocab_inv = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    
    train_dataset = OCRDataset(TRAIN_LABELS, vocab)
    val_dataset = OCRDataset(VAL_LABELS, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CRNN(vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    best_val_crr = -1.0
    epochs_no_improve = 0
    max_epochs = 30
    
    for epoch in range(1, max_epochs + 1):
        print(f"GPU memory before epoch: {torch.cuda.memory_allocated()/1e6:.1f} MB")
        torch.cuda.synchronize()
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, targets, target_lengths, _) in enumerate(train_loader):
            images = images.to(device)
            if images.device.type != 'cuda':
                print("WARNING: images not on GPU!")
                break
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, GPU mem: {torch.cuda.memory_allocated()/1e6:.1f} MB")
                
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            outputs = model(images) # (W, batch, vocab_size)
            input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long, device=device)
            
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_dataset)
        
        model.eval()
        val_preds = []
        val_targets = []
        sample_preds = []
        sample_gts = []
        
        with torch.no_grad():
            for images, targets, target_lengths, labels in val_loader:
                images = images.to(device)
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
        
        # STEP 5 — After EVERY epoch print
        print(f"Epoch {epoch}/{max_epochs}")
        print(f"Train loss: {train_loss:.4f} | Val CRR: {val_crr:.2f}% | LR: {current_lr:.6f}")
        for gt, pred in zip(sample_gts, sample_preds):
            print(f"GT: {gt}  PRED: {pred}")
        print("-" * 40)
        
        # STEP 6 — Save best model
        if val_crr > best_val_crr:
            best_val_crr = val_crr
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_model_v3.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"NEW BEST: {val_crr:.2f}%")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 5:
                print("Early stopping triggered.")
                break

if __name__ == '__main__':
    train()
