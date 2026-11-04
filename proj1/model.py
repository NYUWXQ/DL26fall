"""Pre-activation Wide ResNet for CIFAR-10, capped at 5M parameters.

Architecture reference: https://github.com/szagoruyko/wide-residual-networks
"""
import torch
from torch import nn
from torch.nn import functional as F


def build_model(config):
    options = dict(config)
    architecture = options.pop('architecture', 'wrn')
    if architecture == 'shake':
        from shakeshake import ShakeShake
        return ShakeShake(**options)
    if architecture == 'se':
        from seresnet import SEResNet
        return SEResNet(**options)
    if architecture == 'wrn':
        return WideResNet(**options)
    raise ValueError(f'Unknown architecture: {architecture}')


class Block(nn.Module):
    def __init__(self, cin, cout, stride, dropout, groups=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, groups=groups, bias=False)
        self.shortcut = nn.Conv2d(cin, cout, 1, stride, bias=False) if cin != cout or stride != 1 else None
        self.dropout = dropout

    def forward(self, x):
        activated = F.relu(self.bn1(x))
        shortcut = x if self.shortcut is None else self.shortcut(activated)
        x = self.conv1(activated)
        x = F.dropout(F.relu(self.bn2(x)), self.dropout, self.training)
        return shortcut + self.conv2(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, width=58, dropout=0.0,
                 stage_widths=None, stage_blocks=None, groups=1, teacher=False):
        super().__init__()
        if depth < 10 or (depth - 4) % 6 or width < 1:
            raise ValueError('depth must be 6n+4, n>=1; width must be positive')
        widths = tuple(stage_widths) if stage_widths is not None else (width, width * 2, width * 4)
        counts = tuple(stage_blocks) if stage_blocks is not None else ((depth - 4) // 6,) * 3
        if len(widths) != 3 or len(counts) != 3 or groups < 1:
            raise ValueError('Specify three stage widths/counts and a positive group count')
        if any(c < 1 or c % groups for c in widths) or any(n < 1 for n in counts):
            raise ValueError('Stage counts must be positive and widths divisible by groups')
        self.stem = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        blocks = []
        cin = 16
        for stage, (cout, count) in enumerate(zip(widths, counts)):
            for block in range(count):
                blocks.append(Block(cin, cout, 2 if stage > 0 and block == 0 else 1, dropout, groups))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(cin)
        self.fc = nn.Linear(cin, 10)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        count = sum(p.numel() for p in self.parameters())
        if count > 5_000_000 and not teacher:
            raise ValueError(f'Model exceeds 5M parameters: {count:,}')

    def forward_features(self, x):
        x = F.relu(self.bn(self.blocks(self.stem(x))))
        return x.mean((2, 3))

    def forward(self, x):
        return self.fc(self.forward_features(x))
