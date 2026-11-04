"""CIFAR SE-ResNet with bottlenecks and pooled projection shortcuts.

Architecture reference:
https://github.com/rigvedrs/Optimised-ResNet-for-CIFAR-10
"""
from torch import nn
from torch.nn import functional as F


class SEBottleneck(nn.Module):
    def __init__(self, cin, width, stride):
        super().__init__()
        cout = 4 * width
        self.residual = nn.Sequential(
            nn.Conv2d(cin, width, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, width, 3, stride, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, cout, 1, bias=False), nn.BatchNorm2d(cout),
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cout, cout // 4, 1), nn.ReLU(),
            nn.Conv2d(cout // 4, cout, 1), nn.Sigmoid(),
        )
        shortcut = []
        if stride != 1:
            shortcut.append(nn.AvgPool2d(2, stride, ceil_mode=True, count_include_pad=False))
        if cin != cout or stride != 1:
            shortcut.extend([nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout)])
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        residual = self.residual(x)
        return F.relu(self.shortcut(x) + residual * self.attention(residual))


class SEResNet(nn.Module):
    def __init__(self, depth=164, width=16):
        super().__init__()
        if depth < 11 or (depth - 2) % 9 or width < 1:
            raise ValueError('SE-ResNet requires depth=9n+2 and a positive width')
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(),
        )
        blocks, cin = [], width
        for stage in range(3):
            channels = width * 2**stage
            for block in range((depth - 2) // 9):
                blocks.append(SEBottleneck(cin, channels, 2 if stage and block == 0 else 1))
                cin = channels * 4
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(cin, 10)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        nn.init.normal_(self.fc.weight, std=1e-3)
        nn.init.zeros_(self.fc.bias)
        count = sum(p.numel() for p in self.parameters())
        if count > 5_000_000:
            raise ValueError(f'Model exceeds 5M parameters: {count:,}')

    def forward_features(self, x):
        return self.blocks(self.stem(x)).mean((2, 3))

    def forward(self, x):
        return self.fc(self.forward_features(x))
