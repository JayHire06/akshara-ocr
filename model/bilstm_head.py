# FILE: /model/bilstm_head.py
import torch
import torch.nn as nn

class BiLSTMHead(nn.Module):
    def __init__(self, input_size: int = 512, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super(BiLSTMHead, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, batch, input_size)
        x, _ = self.rnn(x)
        # output shape: (seq_len, batch, hidden_size * 2)
        return x
