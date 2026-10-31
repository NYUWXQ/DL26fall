"""Parameter-budgeted residual networks for 32x32 images.
面向 32x32 图像的、满足参数预算的残差网络。

The assignment caps the model at 5,000,000 parameters, so every factory
below accepts a width multiplier and exposes `n_params()`. The default
configuration is a pre-activation wide ResNet sized to fit the budget.
作业限制参数量 ≤5,000,000，因此每个工厂函数都接受宽度系数并提供 n_params()。
默认配置是恰好塞进预算的预激活宽残差网络。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PARAM_BUDGET = 5_000_000  # hard cap from project1.pdf / 作业硬上限


def conv_bn(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    """3x3 conv + BatchNorm + ReLU. / 3x3 卷积 + BN + ReLU。"""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class PreActBlock(nn.Module):
    """Pre-activation residual block: x + F(BN-ReLU-conv-BN-ReLU-conv).
    预激活残差块：x + F(BN-ReLU-conv-BN-ReLU-conv)。"""

    def __init__(self, cin: int, cout: int, stride: int = 1, drop: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.drop = drop
        # 1x1 projection only when the shape changes. / 仅形状变化时用 1x1 投影。
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Conv2d(cin, cout, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x), inplace=True)
        pre = x
        if self.short is not None:
            pre = self.short(out)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        if self.drop > 0:
            out = F.dropout(out, self.drop, self.training)
        out = self.conv2(out)
        return out + pre


class PreActResNet(nn.Module):
    """Configurable pre-activation ResNet for CIFAR-style 32x32 inputs.
    可配置的预激活 ResNet，面向 CIFAR 风格 32x32 输入。"""

    def __init__(self, depth: int = 16, width: float = 4.0, num_classes: int = 10, drop: float = 0.0):
        super().__init__()
        # Wide-ResNet convention: (depth - 4) must divide into 3 equal stages.
        # 宽残差网络惯例：(depth-4) 必须能被 3 个阶段均分。
        if (depth - 4) % 6 != 0:
            raise ValueError(f"depth must satisfy (depth-4) % 6 == 0, got {depth}")
        n_blocks = (depth - 4) // 6
        widths = [int(16 * width), int(32 * width), int(64 * width)]
        self.stem = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
        layers = []
        cin = widths[0]
        for i, w in enumerate(widths):
            for b in range(n_blocks):
                stride = 2 if (i > 0 and b == 0) else 1
                layers.append(PreActBlock(cin, w, stride=stride, drop=drop))
                cin = w
        self.blocks = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(cin)
        self.head = nn.Linear(cin, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def build_model(name: str = "preact16x4", num_classes: int = 10, drop: float = 0.0) -> nn.Module:
    """Factory by name. Names encode depth x width, e.g. preact16x4.
    按名称构建模型，名称编码 深度x宽度，如 preact16x4。"""
    import re
    m = re.fullmatch(r"preact(\d+)x([\d.]+)", name)
    if not m:
        raise ValueError("unknown model name: %s" % name)
    return PreActResNet(depth=int(m.group(1)), width=float(m.group(2)), num_classes=num_classes, drop=drop)


def n_params(model: nn.Module) -> int:
    """Count trainable parameters. / 统计可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_budget(model: nn.Module, budget: int = PARAM_BUDGET) -> int:
    """Raise if the model breaks the hard parameter cap; return the count.
    若超出硬上限则报错，否则返回参数量。"""
    n = n_params(model)
    if n > budget:
        raise ValueError("model has %d params > budget %d" % (n, budget))
    return n


if __name__ == "__main__":
    # Quick budget table for the report. / 供报告使用的预算表。
    for name in ["preact16x4", "preact16x6", "preact22x4", "preact28x2", "preact16x8"]:
        try:
            mdl = build_model(name)
            print(f"{name:14s} params={n_params(mdl):>9,d}  ok={n_params(mdl) <= PARAM_BUDGET}")
        except ValueError as e:
            print(f"{name:14s} ERROR {e}")
