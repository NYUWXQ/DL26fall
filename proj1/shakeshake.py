"""CIFAR Shake-Shake, ported from Xavier Gastaldi's Torch implementation.

Source: https://github.com/xgastaldi/shake-shake
Copyright (c) 2017, Xavier Gastaldi. See licenses/shake-shake.txt.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Shake(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, training):
        ctx.training = training
        alpha = torch.rand(len(left), 1, 1, 1, device=left.device) if training else 0.5
        return alpha * left + (1 - alpha) * right

    @staticmethod
    def backward(ctx, gradient):
        beta = torch.rand(len(gradient), 1, 1, 1, device=gradient.device) if ctx.training else 0.5
        return beta * gradient, (1 - beta) * gradient, None


class ShakeBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        def branch():
            return nn.Sequential(
                nn.ReLU(), nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout))
        self.left, self.right = branch(), branch()
        self.stride = stride
        self.project = cin != cout or stride != 1
        if self.project:
            self.skip1 = nn.Conv2d(cin, cout // 2, 1, bias=False)
            self.skip2 = nn.Conv2d(cin, cout // 2, 1, bias=False)
            self.skip_bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        shortcut = x
        if self.project:
            activated = F.relu(x)
            first = self.skip1(F.avg_pool2d(activated, 1, self.stride))
            shifted = F.pad(activated, (1, -1, 1, -1))
            second = self.skip2(F.avg_pool2d(shifted, 1, self.stride))
            shortcut = self.skip_bn(torch.cat((first, second), dim=1))
        return shortcut + Shake.apply(self.left(x), self.right(x), self.training)


class ShakeShake(nn.Module):
    def __init__(self, depth=26, width=32):
        super().__init__()
        if depth < 8 or (depth - 2) % 6 or width < 2 or width % 2:
            raise ValueError('Shake-Shake requires depth=6n+2 and a positive even width')
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16))
        blocks, cin = [], 16
        for stage, cout in enumerate((width, width * 2, width * 4)):
            for block in range((depth - 2) // 6):
                blocks.append(ShakeBlock(cin, cout, 2 if stage and block == 0 else 1))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(cin, 10)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)
        count = sum(parameter.numel() for parameter in self.parameters())
        if count > 5_000_000:
            raise ValueError(f'Model exceeds 5M parameters: {count:,}')

    def forward(self, x):
        return self.fc(F.relu(self.blocks(self.stem(x))).mean((2, 3)))
