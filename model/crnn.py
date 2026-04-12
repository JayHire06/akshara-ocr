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

import torch.nn.functional as F

class STN(nn.Module):
    """Spatial Transformer Network to auto-straighten curved text organically."""
    def __init__(self, in_channels=1):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(True),
            nn.Linear(128, 6)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        x_scaled = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        xs = self.localization(x_scaled)
        xs = xs.view(-1, 64 * 8 * 8)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

class MobileNetBlock(nn.Module):
    """Depthwise Separable Convolution replacing standard Conv2d to shrink ONNX footprint."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(MobileNetBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, stride=stride, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))

class CRNNv6(nn.Module):
    """V6 Architecture: STN + MobileNet CNN + BiLSTM"""
    def __init__(self, vocab_size: int, config: dict):
        super(CRNNv6, self).__init__()
        self.in_channels = config.get('in_channels', 1)
        self.stn = STN(in_channels=self.in_channels)
        self.cnn = nn.Sequential(
            MobileNetBlock(self.in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MobileNetBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MobileNetBlock(128, 256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            MobileNetBlock(256, 512),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            MobileNetBlock(512, 512),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        h, layers, drop = config.get('lstm_hidden', 256), config.get('lstm_layers', 2), config.get('lstm_dropout', 0.3)
        self.rnn = nn.LSTM(512, h, num_layers=layers, bidirectional=True, dropout=drop, batch_first=False)
        self.fc = nn.Linear(h * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stn(x)
        x = self.cnn(x).squeeze(2).permute(2, 0, 1)
        x, _ = self.rnn(x)
        return nn.functional.log_softmax(self.fc(x), dim=2)
