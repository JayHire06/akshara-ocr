# FILE: /model/crnn.py
import torch
import torch.nn as nn
from model.cnn_backbone import CNNBackbone
from model.bilstm_head import BiLSTMHead

class CRNN(nn.Module):
    def __init__(self, vocab_size: int, config: dict):
        super(CRNN, self).__init__()
        self.vocab_size = vocab_size
        self.config = config
        
        self.cnn = CNNBackbone(in_channels=config.get('in_channels', 1))
        
        lstm_input_size = config.get('lstm_input_size', 512)
        lstm_hidden_size = config.get('lstm_hidden_size', 256)
        lstm_num_layers = config.get('lstm_num_layers', 2)
        lstm_dropout = config.get('lstm_dropout', 0.3)
        
        self.rnn = BiLSTMHead(
            input_size=lstm_input_size, 
            hidden_size=lstm_hidden_size, 
            num_layers=lstm_num_layers, 
            dropout=lstm_dropout
        )
        
        # Output: Linear(512 -> vocab_size), then LogSoftmax along vocab dim
        self.linear = nn.Linear(lstm_hidden_size * 2, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 32, W)
        conv_out = self.cnn(x) # (W/16, batch, 512)
        rnn_out = self.rnn(conv_out) # (W/16, batch, 512)
        linear_out = self.linear(rnn_out) # (W/16, batch, vocab_size)
        output = self.log_softmax(linear_out) # (W/16, batch, vocab_size)
        return output
