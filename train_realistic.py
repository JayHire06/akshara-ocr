import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb
import numpy as np
from PIL import Image

os.environ["WANDB_MODE"] = "offline"

root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root_dir)

from data.dataset import OCRDataset
from model.crnn import CRNN
from model.ctc_decoder import CTCDecoder
from preprocess.pipeline import preprocess

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
    print("Initializing Realistic Model Training Sequence")
    
    train_dir = os.path.join(root_dir, 'data', 'realistic_synthetic', 'train')
    train_csv = os.path.join(train_dir, 'labels.csv')
    
    # Needs to capture all potential characters. Ideally uses main vocab.json but let's build from labels just in case
    vocab_path = os.path.join(root_dir, 'vocab_realistic.json') 
    
    print("Loading Realistic Datasets...")
    # Passing the full vocab path, it builds it if absent
    train_dataset = OCRDataset(train_dir, train_csv, vocab_path=os.path.join(root_dir, 'vocab.json'))
    
    vocab_dict = train_dataset.vocab
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
        'save_dir': os.path.join(root_dir, 'model', 'checkpoints_realistic_v2')
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on: {device}")
    
    model = CRNN(vocab_size=vocab_size, config=config).to(device)
    criterion = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    decoder = CTCDecoder(vocab=vocab_list, blank_id=blank_id)
    
    # Get 10 images from real test dir
    real_test_dir = os.path.join(root_dir, 'data', 'real_test', 'hindi')
    gt_file = os.path.join(real_test_dir, 'ground_truth.txt')
    
    real_test_cases = []
    if os.path.exists(gt_file):
        with open(gt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[:10]: # Top 10 only
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    real_test_cases.append((os.path.join(real_test_dir, parts[0]), parts[1]))
                    
    print(f"Loaded {len(real_test_cases)} Real-World evaluation images.")
    
    wandb.init(project='akshara-ocr', name="realistic-retrain-run", config=config)
    
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
        for batch_idx, (images, targets, target_lengths) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long).to(device)
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_loss += loss.item()
            wandb.log({'train_loss_step': loss.item()})
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        print(f"\nEpoch {epoch} Completed. Avg Train Loss = {avg_train_loss:.4f}")
        wandb.log({'epoch': epoch, 'train_loss_epoch': avg_train_loss})
        
        # Validation on Real Images directly
        print("\n--- Running Custom Real-World E2E Evaluation ---")
        model.eval()
        with torch.no_grad():
            for img_path, gt in real_test_cases:
                try:
                    pil_img = Image.open(img_path).convert('RGB')
                    # Preprocess extracts lines natively mimicking the prod pipeline
                    line_images = preprocess(pil_img, language='hi')
                    
                    preds = []
                    for li in line_images:
                        arr = np.array(li).astype(np.float32)
                        if arr.max() > 1.0:
                            arr = arr / 255.0
                        
                        arr_t = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 32, W)
                        out = model(arr_t)
                        decoded = decoder.decode_best_path(out)
                        if decoded:
                            preds.append(decoded[0])
                            
                    pred_str = " ".join(preds)
                    print(f"GT  : {gt}")
                    print(f"Pred: {pred_str}")
                    print("-" * 30)
                    
                except Exception as e:
                    print(f"Eval Error on {os.path.basename(img_path)}: {e}")
                    
        # Always checkpoint every epoch
        ckpt_path = os.path.join(config['save_dir'], f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
        }, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}\n")

if __name__ == "__main__":
    start_training()
