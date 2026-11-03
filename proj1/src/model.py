"""
Custom Residual Network Implementation from Scratch (PyTorch).
Implements the exact mathematical formulation from Project 1:
    y = ReLU(S(x) + F(x))
where:
  - F(x) = BN(Conv_{F_i x F_i}(ReLU(BN(Conv_{F_i x F_i}(x)))))
  - S(x) = x (if dimensions match) or BN(Conv_{K_i x K_i, stride=s}(x))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.model_configs import ResNetSpec


class ResidualUnit(nn.Module):
    """
    Single residual block computing ReLU(S(x) + F(x)).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        conv_kernel: int = 3,
        skip_kernel: int = 1,
    ) -> None:
        super().__init__()
        conv_pad = conv_kernel // 2

        # Residual branch F(x): Conv -> BN -> ReLU -> Conv -> BN
        self.residual_branch = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=conv_kernel,
                stride=stride,
                padding=conv_pad,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=conv_kernel,
                stride=1,
                padding=conv_pad,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        # Skip connection S(x): Identity or K_i x K_i projection shortcut
        if stride != 1 or in_channels != out_channels:
            skip_pad = skip_kernel // 2
            self.skip_connection = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=skip_kernel,
                    stride=stride,
                    padding=skip_pad,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.skip_connection(x) + self.residual_branch(x), inplace=True)


class CustomResNet(nn.Module):
    """
    Full N-Stage Residual Network parameterized by ResNetSpec (N, B_i, C_i, F_i, K_i, P).
    """

    def __init__(self, spec: ResNetSpec) -> None:
        super().__init__()
        self.spec = spec
        self.current_channels = spec.stem_channels

        # Initial feature-extraction stem preserving 32x32 spatial resolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, spec.stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(spec.stem_channels),
            nn.ReLU(inplace=True),
        )

        # Stack N residual stages
        stages = []
        for idx in range(len(spec.channels)):
            stage_stride = 1 if idx == 0 else 2
            stages.append(
                self._build_stage(
                    out_channels=spec.channels[idx],
                    num_blocks=spec.blocks_per_stage[idx],
                    first_stride=stage_stride,
                    conv_kernel=spec.conv_kernels[idx],
                    skip_kernel=spec.skip_kernels[idx],
                )
            )
        self.residual_stages = nn.Sequential(*stages)

        # P x P Average Pooling + Linear Classification Head
        self.pool_size = spec.pool_size
        self.dropout = nn.Dropout(p=spec.dropout_rate) if spec.dropout_rate > 0 else nn.Identity()

        final_map_width = 32 // (2 ** (len(spec.channels) - 1))
        pooled_width = max(1, final_map_width // spec.pool_size)
        fc_input_dim = spec.channels[-1] * (pooled_width ** 2)
        self.classifier = nn.Linear(fc_input_dim, spec.num_classes)

        self._init_weights()

    def _build_stage(
        self,
        out_channels: int,
        num_blocks: int,
        first_stride: int,
        conv_kernel: int,
        skip_kernel: int,
    ) -> nn.Sequential:
        blocks = []
        for b_idx in range(num_blocks):
            stride = first_stride if b_idx == 0 else 1
            blocks.append(
                ResidualUnit(
                    in_channels=self.current_channels,
                    out_channels=out_channels,
                    stride=stride,
                    conv_kernel=conv_kernel,
                    skip_kernel=skip_kernel,
                )
            )
            self.current_channels = out_channels
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stem(x)
        features = self.residual_stages(features)
        pooled = F.avg_pool2d(features, kernel_size=self.pool_size)
        flattened = torch.flatten(pooled, start_dim=1)
        return self.classifier(self.dropout(flattened))


def count_trainable_parameters(model: nn.Module) -> int:
    """Returns the exact number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def audit_model_parameters(model: CustomResNet, max_budget: int = 5_000_000, run_summary: bool = True) -> int:
    """
    Prints the total trainable parameter count, verifies the <= 5,000,000 constraint,
    and optionally invokes torchsummary.summary(model, (3, 32, 32)).
    """
    total_params = count_trainable_parameters(model)
    pct = 100.0 * total_params / max_budget
    print("=" * 76)
    print(f"MODEL PARAMETER AUDIT: {model.spec.name}")
    print(f"  Stages (N)         : {len(model.spec.channels)}")
    print(f"  Channels (C_i)     : {model.spec.channels}")
    print(f"  Blocks (B_i)       : {model.spec.blocks_per_stage}")
    print(f"  Conv Kernels (F_i) : {model.spec.conv_kernels}")
    print(f"  Skip Kernels (K_i) : {model.spec.skip_kernels}")
    print(f"  Avg Pool Size (P)  : {model.spec.pool_size}")
    print(f"  Trainable Params   : {total_params:,} / {max_budget:,} ({pct:.2f}% of budget)")
    print("=" * 76)
    assert total_params <= max_budget, (
        f"Constraint violation: {total_params:,} exceeds {max_budget:,} parameters!"
    )
    if run_summary:
        try:
            from torchsummary import summary
            dev = "cuda" if next(model.parameters()).is_cuda else "cpu"
            summary(model, input_size=(3, 32, 32), device=dev)
        except Exception as e:
            print(f"(Note: torchsummary skipped: {e})")
    return total_params
