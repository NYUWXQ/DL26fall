"""CIFAR PyramidNet with pixel-wise ShakeDrop, adapted from the authors' code.

https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/PyramidNet.py
https://github.com/imenurok/ShakeDrop/blob/master/pyramidnet.lua
https://github.com/imenurok/ShakeDrop/blob/master/ShakeDrop.lua
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ShakeDrop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, probability, training):
        ctx.training = training
        ctx.probability = probability
        if not training:
            return x * (1 - probability)
        gate = (torch.rand((), device=x.device) >= probability).to(x.dtype)
        ctx.save_for_backward(gate)
        return x * (gate + (1 - gate) * torch.empty_like(x).uniform_(-1, 1))

    @staticmethod
    def backward(ctx, gradient):
        if not ctx.training:
            return gradient * (1 - ctx.probability), None, None
        gate, = ctx.saved_tensors
        scale = gate + (1 - gate) * torch.rand_like(gradient)
        return gradient * scale, None, None


class Block(nn.Module):
    def __init__(self, cin, cout, stride, probability):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)
        self.stride = stride
        self.probability = probability

    def forward(self, x):
        residual = self.conv1(self.bn1(x))
        residual = self.bn3(self.conv2(F.relu(self.bn2(residual))))
        residual = ShakeDrop.apply(residual, self.probability, self.training)
        shortcut = F.avg_pool2d(x, 2, 2, ceil_mode=True) if self.stride == 2 else x
        shortcut = F.pad(shortcut, (0, 0, 0, 0, 0, residual.shape[1] - shortcut.shape[1]))
        return shortcut + residual


class PyramidNet(nn.Module):
    def __init__(self, depth=110, alpha=96, death_rate=0.5):
        super().__init__()
        if depth < 8 or (depth - 2) % 6 or alpha < 1:
            raise ValueError('Basic-block PyramidNet requires depth=6n+2 and positive alpha')
        if not math.isfinite(death_rate) or not 0 <= death_rate <= 1:
            raise ValueError('ShakeDrop death rate must be between zero and one')
        n = (depth - 2) // 6
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16))
        blocks, cin = [], 16
        for i in range(3 * n):
            cout = round(16 + alpha * (i + 1) / (3 * n))
            stride = 2 if i in (n, 2 * n) else 1
            blocks.append(Block(cin, cout, stride, death_rate * (i + 1) / (3 * n)))
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(cin)
        self.fc = nn.Linear(cin, 10)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)
        count = sum(p.numel() for p in self.parameters())
        if count > 5_000_000:
            raise ValueError(f'Model exceeds 5M parameters: {count:,}')

    def forward_features(self, x):
        return F.relu(self.bn(self.blocks(self.stem(x)))).mean((2, 3))

    def forward(self, x):
        return self.fc(self.forward_features(x))
