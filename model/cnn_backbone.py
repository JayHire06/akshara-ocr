# FILE: /model/cnn_backbone.py
import torch
import torch.nn as nn

class CNNBackbone(nn.Module):
    def __init__(self, in_channels: int = 1):
        super(CNNBackbone, self).__init__()
        # 5 blocks of (Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d)
        # Channels: 64, 128, 256, 256, 512
        # Pool: (2,2), (2,2), (2,1), (2,1), (2,1)
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch, 1, 32, W)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        # After CNN: feature map shape (batch, 512, 1, W/16) due to width pools being 2x2.
        # Squeeze height dim (dimension 2)
        x = x.squeeze(2) # Shape: (batch, 512, seq_len)
        
        # Permute to (W/16, batch, 512) for RNN input
        x = x.permute(2, 0, 1) # Shape: (seq_len, batch, 512)
        
        return x
