# FILE: /model/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from model.crnn import CRNN
from model.evaluate import evaluate_crr

def train(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    vocab: list
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    criterion = nn.CTCLoss(blank=config.get('blank_id', 0), reduction='mean', zero_infinity=True)
    
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.get('lr_factor', 0.5), patience=config.get('lr_patience', 3)
    )
    
    epochs = config.get('epochs', 50)
    clip_grad_norm = config.get('clip_grad_norm', 5.0)
    save_dir = config.get('save_dir', 'model/checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    wandb.init(project='akshara-ocr', config=config)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, targets, target_lengths in pbar:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            
            output = model(images)  # (seq_len, batch, vocab_size)
            seq_len, batch_size, _ = output.shape
            
            input_lengths = torch.full(size=(batch_size,), fill_value=seq_len, dtype=torch.long, device=device)
            
            loss = criterion(output, targets, input_lengths, target_lengths)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            optimizer.step()
            train_loss += loss.item() * batch_size
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        val_loss, val_crr = evaluate_crr(model, val_loader, criterion, device, vocab, blank_id=config.get('blank_id', 0))
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val CRR = {val_crr:.4f}")
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_crr': val_crr,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        save_interval = config.get('save_interval', 5)
        if epoch % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_crr': val_crr
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
    wandb.finish()
