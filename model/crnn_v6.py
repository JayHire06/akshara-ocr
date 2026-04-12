import torch
import torch.nn as nn
import torch.nn.functional as F

class STN(nn.Module):
    """
    Spatial Transformer Network (STN) to auto-straighten curved or slanted text organically.
    """
    def __init__(self, in_channels=1):
        super(STN, self).__init__()
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128), # Assuming input is (1, 32, 32) at minimum scaled, STN is dynamically resizing
            nn.ReLU(True),
            nn.Linear(128, 6)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # Dynamically interpolate to 32x32 for stable affine prediction, regardless of input length
        x_scaled = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        xs = self.localization(x_scaled)
        xs = xs.view(-1, 64 * 8 * 8)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x


class MobileNetBlock(nn.Module):
    """
    Depthwise Separable Convolution replacing standard Conv2d to shrink ONNX footprint.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(MobileNetBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, stride=stride, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class CRNNv6(nn.Module):
    """
    V6 Architecture: STN + MobileNet CNN + BiLSTM
    """
    def __init__(self, vocab_size: int, config: dict):
        super(CRNNv6, self).__init__()
        self.in_channels = config.get('in_channels', 1)
        self.stn = STN(in_channels=self.in_channels)
        
        # Swapped conventional CNN backbone with Depthwise mapping
        self.cnn = nn.Sequential(
            MobileNetBlock(self.in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MobileNetBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MobileNetBlock(128, 256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            MobileNetBlock(256, 512),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        h = config.get('lstm_hidden', 256)
        layers = config.get('lstm_layers', 2)
        drop = config.get('lstm_dropout', 0.3)
        
        self.rnn = nn.LSTM(
            512, h,
            num_layers=layers,
            bidirectional=True,
            dropout=drop,
            batch_first=False
        )
        self.fc = nn.Linear(h * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-process visually warped text natively
        x = self.stn(x)
        
        x = self.cnn(x)          # (B, 512, 1, W')
        x = x.squeeze(2)         # (B, 512, W')
        x = x.permute(2, 0, 1)   # (W', B, 512)
        x, _ = self.rnn(x)       # (W', B, H*2)
        x = self.fc(x)           # (W', B, vocab)
        return nn.functional.log_softmax(x, dim=2)
