import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalCTCLoss(nn.Module):
    """
    Focal CTC Loss: applies a scaling factor (1 - exp(-loss))**self.gamma to standard CTC.
    This forces the gradient to pay 10x more attention to complex/rare classes missing in standard data, 
    effectively bypassing 'class unbalancing' inherently natively inside CTC paths.
    """
    def __init__(self, blank_id=0, gamma=2.0, alpha=1.0, zero_infinity=True):
        super(FocalCTCLoss, self).__init__()
        self.ctc = nn.CTCLoss(blank=blank_id, reduction='none', zero_infinity=zero_infinity)
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # 1. Calculate Standard CTC Loss but UNREDUCED to get per-sequence matrices
        loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        
        # 2. Extract probability distribution p = exp(-loss)
        p = torch.exp(-loss)
        
        # 3. Apply Focal weight map -> alpha * (1-p)**gamma * loss
        focal_loss = self.alpha * ((1 - p) ** self.gamma) * loss
        
        return focal_loss.mean()
