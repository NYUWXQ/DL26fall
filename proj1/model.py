"""Pre-activation Wide ResNet for CIFAR-10, capped at 5M parameters.

Architecture reference: https://github.com/szagoruyko/wide-residual-networks
"""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride, dropout):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.shortcut = nn.Conv2d(cin, cout, 1, stride, bias=False) if cin != cout or stride != 1 else None
        self.dropout = dropout

    def forward(self, x):
        activated = F.relu(self.bn1(x))
        shortcut = x if self.shortcut is None else self.shortcut(activated)
        x = self.conv1(activated)
        x = F.dropout(F.relu(self.bn2(x)), self.dropout, self.training)
        return shortcut + self.conv2(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, width=58, dropout=0.0):
        super().__init__()
        if depth < 10 or (depth - 4) % 6 or width < 1:
            raise ValueError('depth must be 6n+4, n>=1; width must be positive')
        self.stem = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        blocks = []
        cin = 16
        for stage, cout in enumerate((width, width * 2, width * 4)):
            for block in range((depth - 4) // 6):
                blocks.append(Block(cin, cout, 2 if stage > 0 and block == 0 else 1, dropout))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(cin)
        self.fc = nn.Linear(cin, 10)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        count = sum(p.numel() for p in self.parameters())
        if count > 5_000_000:
            raise ValueError(f'Model exceeds 5M parameters: {count:,}')

    def forward(self, x):
        x = F.relu(self.bn(self.blocks(self.stem(x))))
        return self.fc(x.mean((2, 3)))
