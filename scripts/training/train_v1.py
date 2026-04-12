import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

os.environ["WANDB_MODE"] = "offline"

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, root_dir)

from data.dataset import OCRDataset
from model.crnn import CRNN
from model.train import train

def collate_fn(batch):
    images = [item['image'] for item in batch]
    labels_encoded = [item['label_encoded'] for item in batch]
    max_width = max(img.shape[2] for img in images)
    
    padded_images = []
    for img in images:
        h = img.shape[1]
        w = img.shape[2]
        if h != 32:
            new_w = max(1, int(w * (32 / h)))
            img = nn.functional.interpolate(img.unsqueeze(0), size=(32, new_w), mode='bilinear', align_corners=False).squeeze(0)
        w = img.shape[2]
        pad_w = max(0, max_width - w)
        padded_img = nn.functional.pad(img, (0, pad_w, 0, 0))
        padded_images.append(padded_img)
        
    actual_max_width = max(img.shape[2] for img in padded_images)
    final_padded_images = [nn.functional.pad(img, (0, actual_max_width - img.shape[2], 0, 0)) for img in padded_images]
    images_tensor = torch.stack(final_padded_images)
    
    target_lengths = torch.tensor([len(lbl) for lbl in labels_encoded], dtype=torch.long)
    targets = torch.cat(labels_encoded)
    return images_tensor, targets, target_lengths

def start_training():
    print("Initializing Native Base V1 Model Training Sequence")
    
    # Utilizing base combined fallback directory mapped originally by data generator scripts
    train_dir = os.path.join(root_dir, 'data', 'combined')
    train_csv = os.path.join(train_dir, 'train_labels.txt')
    val_csv = os.path.join(train_dir, 'val_labels.txt')
    vocab_path = os.path.join(root_dir, 'vocab.json')
    
    print("Loading Base Datasets...")
    train_dataset = OCRDataset(train_dir, train_csv, vocab_path=vocab_path, is_v5_format=True)
    val_dataset = OCRDataset(train_dir, val_csv, vocab_path=vocab_path, is_v5_format=True)
    
    # We cap dataset artificially because base generic pipelines weren't tuned for millions of items
    train_subset_size = min(50_000, len(train_dataset)) 
    val_subset_size = min(5_000, len(val_dataset)) 
    print(f"Loading generic subsets... train: {train_subset_size}, val: {val_subset_size}")
    
    train_dataset = torch.utils.data.Subset(train_dataset, range(train_subset_size))
    val_dataset = torch.utils.data.Subset(val_dataset, range(val_subset_size))
    
    vocab_dict = val_dataset.dataset.vocab
    vocab_size = len(vocab_dict)
    
    vocab_list = [""] * vocab_size
    for char, idx in vocab_dict.items():
        vocab_list[idx] = char
        
    blank_id = vocab_dict.get('<PAD>', 0)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    config = {
        'in_channels': 1,
        'lstm_input_size': 512,
        'lstm_hidden_size': 256,
        'lstm_num_layers': 2,
        'lstm_dropout': 0.3,
        'blank_id': blank_id,
        'lr': 1e-3,
        'epochs': 10,
        'save_dir': os.path.join(root_dir, 'model', 'checkpoints_v1')
    }
    
    print("Loading V1 Native Execution Matrix")
    model = CRNN(vocab_size=vocab_size, config=config)
    os.makedirs(config['save_dir'], exist_ok=True)
    
    train(model, train_loader, val_loader, config, vocab_list)
    print("V1 Benchmark Completed.")

if __name__ == "__main__":
    start_training()
