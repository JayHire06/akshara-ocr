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

def build_vocab(labels_csv, vocab_path="vocab.json"):
    """Extracts unique characters from dataset and saves vocabulary."""
    chars = set()
    with open(labels_csv, "r", encoding="utf-8") as f:
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
    def __init__(self, data_dir, labels_csv, vocab_path="vocab.json", transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        with open(labels_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)
                
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            self.vocab = build_vocab(labels_csv, vocab_path)
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        language = sample['language']
        # Looking sub-directories based on language
        img_path = os.path.join(self.data_dir, language, sample['filename'])
        
        image = Image.open(img_path).convert("L")
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

if __name__ == '__main__':
    # Example usage
    # dataset = OCRDataset("synthetic", "synthetic/labels.csv", "vocab.json")
    # if len(dataset) > 0:
    #     print("Sample 0:", dataset[0])
    pass
