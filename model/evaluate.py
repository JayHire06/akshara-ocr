# FILE: /model/evaluate.py
import torch
import torch.nn as nn
from typing import Tuple, List
from model.ctc_decoder import CTCDecoder

def compute_edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return compute_edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
        
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def evaluate_crr(
    model: nn.Module,
    val_loader,
    criterion: nn.CTCLoss,
    device: torch.device,
    vocab: List[str],
    blank_id: int = 0
) -> Tuple[float, float]:
    
    model.eval()
    val_loss = 0.0
    decoder = CTCDecoder(vocab=vocab, blank_id=blank_id)
    
    total_chars = 0
    total_edit_distance = 0
    
    with torch.no_grad():
        for images, targets, target_lengths in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            output = model(images)
            seq_len, batch_size, _ = output.shape
            input_lengths = torch.full(size=(batch_size,), fill_value=seq_len, dtype=torch.long, device=device)
            
            loss = criterion(output, targets, input_lengths, target_lengths)
            val_loss += loss.item() * batch_size
            
            # Decode predictions
            decoded_preds = decoder.decode_best_path(output)
            
            # Decode targets
            target_idx = 0
            for i in range(batch_size):
                tgt_len = target_lengths[i].item()
                tgt_tokens = targets[target_idx:target_idx+tgt_len].tolist()
                target_idx += tgt_len
                
                gt_str = decoder.convert_to_string(tgt_tokens)
                pred_str = decoded_preds[i]
                
                total_chars += len(gt_str)
                total_edit_distance += compute_edit_distance(pred_str, gt_str)
                
    val_loss /= len(val_loader.dataset)
    
    # Character Recognition Rate (CRR) = 1 - (Edit Distance / Total Characters)
    crr = max(0.0, 1.0 - (total_edit_distance / total_chars)) if total_chars > 0 else 0.0
    
    return val_loss, crr
