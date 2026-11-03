"""Densely connected CNN for the CIFAR-10 variant challenge (DenseNet-BC).

Architecture: Huang et al., "Densely Connected Convolutional Networks",
arXiv:1608.06993. Configuration DenseNet-BC (L=100, k=12): three dense blocks of
16 layers, growth rate 12, bottleneck 4k, compression 0.5 -> 0.8M parameters,
paper CIFAR-10+ error 4.51%.

架构：Huang et al., arXiv:1608.06993。DenseNet-BC (L=100, k=12)：三个 dense block
各 16 层，growth rate 12，bottleneck 4k，compression 0.5 -> 0.8M 参数，
论文 CIFAR-10+ 错误率 4.51%。

Why this family: the measured leaderboard lever was architectural diversity
(+0.3 per added family), not epochs, so a structurally different small network is
more valuable to the fusion than another seed of the same family.
为什么用这一族：实测的杠杆是架构多样性（每加一族 +0.3），而不是 epoch，所以一个
结构不同的小网络比同族的又一个 seed 更有价值。
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    """BN-ReLU-1x1(4k)-BN-ReLU-3x3(k), concatenated onto the input.
    BN-ReLU-1x1(4k)-BN-ReLU-3x3(k)，输出与输入拼接。"""

    def __init__(self, cin: int, growth: int, bn_size: int = 4, drop: float = 0.0):
        super().__init__()
        inter = bn_size * growth
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, inter, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter)
        self.conv2 = nn.Conv2d(inter, growth, 3, padding=1, bias=False)
        self.drop = drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        if self.drop > 0:
            out = F.dropout(out, self.drop, self.training)
        return torch.cat([x, out], 1)


class Transition(nn.Module):
    """BN-ReLU-1x1(compress)-avgpool2. / BN-ReLU-1x1(压缩)-平均池化。"""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(cin)
        self.conv = nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(self.conv(F.relu(self.bn(x), inplace=True)), 2)


class DenseNetBC(nn.Module):
    """DenseNet-BC for 32x32 inputs. / 面向 32x32 输入的 DenseNet-BC。"""

    def __init__(self, layers: Sequence[int] = (16, 16, 16), growth: int = 12,
                 compression: float = 0.5, num_classes: int = 10, drop: float = 0.0):
        super().__init__()
        self.growth = growth
        cin = 2 * growth
        self.stem = nn.Conv2d(3, cin, 3, padding=1, bias=False)
        blocks = []
        for i, n in enumerate(layers):
            for _ in range(n):
                blocks.append(DenseLayer(cin, growth, drop=drop))
                cin += growth
            if i != len(layers) - 1:  # no transition after the last block / 最后一块后不接 transition
                cout = int(cin * compression)
                blocks.append(Transition(cin, cout))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(cin)
        self.head = nn.Linear(cin, num_classes)
        # Same init policy as the rest of the project. / 与项目其余模型一致的初始化。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(x))
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)
