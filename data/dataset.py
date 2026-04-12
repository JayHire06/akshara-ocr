# data/dataset.py
import os
import json
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def build_vocab(labels_csv, vocab_path="vocab.json", is_v5_format=False):
    """Extracts unique characters from dataset and saves vocabulary."""
    chars = set()
    with open(labels_csv, "r", encoding="utf-8") as f:
        if is_v5_format:
            for line in f:
                if "|" in line:
                    _, label = line.strip().split("|", 1)
                    chars.update(label)
        else:
            reader = csv.DictReader(f)
            for row in reader:
                for char in row['text']:
                    chars.add(char)
                
    vocab = {char: idx for idx, char in enumerate(sorted(list(chars)), start=1)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = len(vocab)
    
    vocab_dir = os.path.dirname(vocab_path)
    if vocab_dir:
        os.makedirs(vocab_dir, exist_ok=True)
        
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
        
    return vocab

class OCRDataset(Dataset):
    """PyTorch Dataset class for loading image+label pairs."""
    def __init__(self, data_dir, labels_csv, vocab_path="vocab.json", transform=None, is_v5_format=False):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.is_v5_format = is_v5_format
        
        with open(labels_csv, "r", encoding="utf-8") as f:
            if is_v5_format:
                for line in f:
                    if "|" in line:
                        path_p, label = line.strip().split("|", 1)
                        if os.path.isabs(path_p):
                            self.samples.append({"filename": path_p, "text": label, "language": "hi"})
                        else:
                            self.samples.append({"filename": os.path.join(data_dir, path_p), "text": label, "language": "hi"})
            else:
                reader = csv.DictReader(f)
                for row in reader:
                    self.samples.append(row)
                
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
            if '<UNK>' not in self.vocab:
                self.vocab['<UNK>'] = len(self.vocab)
        else:
            self.vocab = build_vocab(labels_csv, vocab_path, is_v5_format)
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        language = sample.get('language', 'hi')
        
        if self.is_v5_format:
            img_path = sample['filename']
        else:
            img_path = os.path.join(self.data_dir, language, sample['filename'])
        
        image = Image.open(img_path).convert("L")
        w, h = image.size
        # Normalize height if v1 generic loader tries hitting weird shapes
        new_w = max(1, int(w * (32 / h))) if h != 32 else w
        image = image.resize((new_w, 32), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(image)).float().unsqueeze(0) / 255.0
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        text = sample['text']
        label_encoded = [self.vocab.get(char, self.vocab['<UNK>']) for char in text]
        label_encoded = torch.tensor(label_encoded, dtype=torch.long)
        
        return {
            'image': img_tensor,
            'label': text,
            'label_encoded': label_encoded,
            'language': language
        }

import cv2
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

class OCRDatasetV6(Dataset):
    """Advanced V6 Dataset employing OpenCV and Albumentations for robust spatial simulated degradation mapping."""
    def __init__(self, labels_file, vocab, is_training=True):
        self.vocab = vocab
        self.is_training = is_training
        self.samples = []
        with open(labels_file, encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    path_p, label = line.strip().split("|", 1)
                    if label: self.samples.append((path_p.strip(), label.strip()))
        
        self.transform = None
        if ALBUMENTATIONS_AVAILABLE and self.is_training:
            self.transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=3, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.4),
                A.ElasticTransform(alpha=1, sigma=30, alpha_affine=20, p=0.3),
                A.GridDistortion(num_steps=4, distort_limit=0.1, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.CoarseDropout(max_holes=5, max_height=4, max_width=8, fill_value=255, p=0.3)
            ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if self.transform: img = self.transform(image=img)['image']
            h, w = img.shape
            new_w = max(1, int(w * 32 / max(h, 1)))
            img_resized = cv2.resize(img, (new_w, 32), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).unsqueeze(0)
        except Exception:
            tensor = torch.zeros(1, 32, 64)
        
        encoded = [self.vocab[c] for c in label if c in self.vocab]
        return tensor, torch.tensor(encoded, dtype=torch.long), label
