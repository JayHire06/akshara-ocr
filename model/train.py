# FILE: /model/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
try:
    import wandb
except (ImportError, TypeError):
    class _DummyWandb:
        def init(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
        def Table(self, *args, **kwargs): return self
        def add_data(self, *args, **kwargs): pass
        def finish(self, *args, **kwargs): pass
    wandb = _DummyWandb()
from tqdm import tqdm
from model.crnn import CRNN
from model.evaluate import evaluate_crr

from model.ctc_decoder import CTCDecoder

def train(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    vocab: list
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on: {device}")
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
    
    start_epoch = 1
    best_val_crr = -1.0
    
    resume_checkpoint = config.get('resume_checkpoint', None)
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_crr = checkpoint.get('val_crr', -1.0)
        print(f"Skipping previous epoch logic. Restarting at Epoch {start_epoch}")
        
    os.makedirs(save_dir, exist_ok=True)
    
    wandb.init(project='akshara-ocr', config=config)
    
    early_stop_patience = config.get('patience', 3)
    epochs_no_improve = 0
    
    decoder = CTCDecoder(vocab=vocab, blank_id=config.get('blank_id', 0))
    global_step = 0
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        
        print(f"\n--- Epoch {epoch}/{epochs} Started ---")
        for batch_idx, (images, targets, target_lengths) in enumerate(train_loader):
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
            global_step += 1
            
            if batch_idx % 200 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
            # W&B Sample Prediction Logging every 1000 steps
            if global_step % 1000 == 0:
                with torch.no_grad():
                    model.eval()
                    try:
                        # Grab predictions for the current batch
                        decoded_preds = decoder.decode_best_path(output)
                        
                        # Decode targets for comparison
                        target_idx = 0
                        gt_strings = []
                        for i in range(batch_size):
                            tgt_len = target_lengths[i].item()
                            tgt_tokens = targets[target_idx:target_idx+tgt_len].tolist()
                            target_idx += tgt_len
                            gt_strings.append(decoder.convert_to_string(tgt_tokens))
                        
                        # Log to W&B Table (limit to 10 for neatness)
                        table = wandb.Table(columns=["Ground Truth", "Predicted"])
                        for gt, pred in zip(gt_strings[:10], decoded_preds[:10]):
                            table.add_data(gt, pred)
                        
                        wandb.log({f"Sample Predictions (Step {global_step})": table})
                    finally:
                        model.train()
                        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        val_loss, val_crr = evaluate_crr(model, val_loader, criterion, device, vocab, blank_id=config.get('blank_id', 0))
        scheduler.step(val_loss)
        
        # Calculate Real Accuracy (20 predictions without <UNK>)
        model.eval()
        no_unk_count = 0
        samples_checked = 0
        with torch.no_grad():
            for val_img, val_tgt, val_tgt_len in val_loader:
                val_img = val_img.to(device)
                val_out = model(val_img)
                val_preds = decoder.decode_best_path(val_out)
                for pred in val_preds:
                    if '<UNK>' not in pred:
                        no_unk_count += 1
                    samples_checked += 1
                    if samples_checked >= 20:
                        break
                if samples_checked >= 20:
                    break
        
        real_accuracy = (no_unk_count / 20.0) * 100.0
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val CRR = {val_crr:.4f}")
        print(f"** Real Accuracy (<UNK> Free): {no_unk_count}/20 ({real_accuracy:.1f}%) **")
        
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_crr': val_crr,
            'real_accuracy': real_accuracy,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Checkpointing and Early Stopping Logic
        if val_crr > best_val_crr:
            best_val_crr = val_crr
            epochs_no_improve = 0
            
            # Best model save
            best_model_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_crr': val_crr
            }, best_model_path)
            print(f"New best CRR! Saved best_model.pth with CRR: {val_crr:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            
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
            
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered! Training halted at epoch {epoch}.")
            break
            
    wandb.finish()
