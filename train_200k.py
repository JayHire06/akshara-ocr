import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

os.environ["WANDB_MODE"] = "offline"

root_dir = os.path.abspath(os.path.dirname(__file__))
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
    print("Initializing 200K Model Training Sequence (Target: < 10% UNK validation)")
    
    train_dir = os.path.join(root_dir, 'data', 'train')
    val_dir = os.path.join(root_dir, 'data', 'val')
    train_csv = os.path.join(train_dir, 'labels.csv')
    val_csv = os.path.join(val_dir, 'labels.csv')
    vocab_path = os.path.join(root_dir, 'vocab.json')
    
    print("Loading Datasets...")
    full_train_dataset = OCRDataset(train_dir, train_csv, vocab_path=vocab_path)
    full_val_dataset = OCRDataset(val_dir, val_csv, vocab_path=vocab_path)
    
    train_subset_size = min(200_000, len(full_train_dataset))
    val_subset_size = min(10_000, len(full_val_dataset)) 
    
    print(f"Slicing Training to {train_subset_size} and Validation to {val_subset_size}...")
    
    # We explicitly seed to ensure determinstic subsets if needed, but range is fine since dataset is shuffled externally
    train_dataset = Subset(full_train_dataset, range(train_subset_size))
    val_dataset = Subset(full_val_dataset, range(val_subset_size))
    
    vocab_dict = full_train_dataset.vocab
    vocab_size = len(vocab_dict)
    
    vocab_list = [""] * vocab_size
    for char, idx in vocab_dict.items():
        vocab_list[idx] = char
        
    blank_id = vocab_dict.get('<PAD>', 0)
    print(f"Vocab size compiled: {vocab_size} classes")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    print("Initializing Native CRNN Model...")
    config = {
        'in_channels': 1,
        'lstm_input_size': 512,
        'lstm_hidden_size': 256,
        'lstm_num_layers': 2,
        'lstm_dropout': 0.3,
        'blank_id': blank_id,
        'lr': 1e-3,
        'lr_factor': 0.5,
        'lr_patience': 3,
        'epochs': 10,
        'patience': 10, # Disabled Early Stopping for 10 epochs
        'save_interval': 2, 
        'clip_grad_norm': 5.0,
        'save_dir': os.path.join(root_dir, 'model', 'checkpoints_200k'),
        'resume_checkpoint': os.path.join(root_dir, 'model', 'checkpoints_200k', 'checkpoint_epoch_6.pth')
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    model = CRNN(vocab_size=vocab_size, config=config)
    
    print(f"Starting 10-epoch pipeline explicitly over 200K...")
    
    try:
        train(model, train_loader, val_loader, config, vocab_list)
        print("Training execution concluded.")
    except Exception as e:
        print(f"Error raised during Training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start_training()
