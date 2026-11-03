"""
Architectural Configuration Definitions for Deep Learning Project 1 (Try 5: Ultra Sub-5M).
Built 100% from scratch based on the Project 1 specification:
  - N:   Number of residual stages (len(channels))
  - B_i: Number of residual blocks in stage i (blocks_per_stage)
  - C_i: Number of channels in stage i (channels)
  - F_i: Spatial kernel size for residual convolutions in stage i (conv_kernels)
  - K_i: Spatial kernel size for projection skip connections in stage i (skip_kernels)
  - P:   Average pooling kernel size before the classification head (pool_size)
"""

from dataclasses import dataclass, asdict
from typing import List, Dict


@dataclass
class ResNetSpec:
    name: str
    stem_channels: int
    channels: List[int]          # C_i for i = 1..N
    blocks_per_stage: List[int]  # B_i for i = 1..N
    conv_kernels: List[int]      # F_i for i = 1..N
    skip_kernels: List[int]      # K_i for i = 1..N
    pool_size: int               # P
    dropout_rate: float = 0.0
    num_classes: int = 10

    def __post_init__(self) -> None:
        n_stages = len(self.channels)
        assert len(self.blocks_per_stage) == n_stages, "blocks_per_stage length must match channels"
        assert len(self.conv_kernels) == n_stages, "conv_kernels length must match channels"
        assert len(self.skip_kernels) == n_stages, "skip_kernels length must match channels"

    def to_dict(self) -> dict:
        return asdict(self)


def get_architecture_registry() -> Dict[str, ResNetSpec]:
    """
    Returns our library of custom, from-scratch ResNet specifications, all strictly
    under the 5,000,000 trainable parameter ceiling.
    """
    return {
        # 1. Baseline 4-Stage Tapered ResNet (4,184,522 parameters — 83.69% of 5M budget)
        "baseline_4stage": ResNetSpec(
            name="baseline_4stage",
            stem_channels=64,
            channels=[64, 128, 192, 256],
            blocks_per_stage=[2, 2, 2, 2],
            conv_kernels=[3, 3, 3, 3],
            skip_kernels=[1, 1, 1, 1],
            pool_size=4,
            dropout_rate=0.0,
        ),
        # 2. Wide 3-Stage ResNet with P=8 Average Pooling (4,773,930 parameters — 95.48% of 5M budget)
        "wide_3stage": ResNetSpec(
            name="wide_3stage",
            stem_channels=64,
            channels=[64, 144, 264],
            blocks_per_stage=[3, 3, 3],
            conv_kernels=[3, 3, 3],
            skip_kernels=[1, 1, 1],
            pool_size=8,
            dropout_rate=0.05,
        ),
        # 3. 3x3 Spatial Skip-Connection Kernel Variant (4,839,882 parameters — 96.80% of 5M budget)
        "skip3x3_4stage": ResNetSpec(
            name="skip3x3_4stage",
            stem_channels=64,
            channels=[64, 128, 192, 256],
            blocks_per_stage=[2, 2, 2, 2],
            conv_kernels=[3, 3, 3, 3],
            skip_kernels=[1, 3, 3, 3],
            pool_size=4,
            dropout_rate=0.05,
        ),
        # 4. Early-Depth Champion ResNet (4,917,450 parameters — 98.35% of 5M budget)
        "champion_sub5m": ResNetSpec(
            name="champion_sub5m",
            stem_channels=64,
            channels=[64, 128, 208, 264],
            blocks_per_stage=[3, 3, 2, 2],
            conv_kernels=[3, 3, 3, 3],
            skip_kernels=[1, 1, 1, 1],
            pool_size=4,
            dropout_rate=0.1,
        ),
        # 5. Try 5 Ultra Sub-5M ResNet (4,977,834 parameters — 99.56% of 5M budget)
        "ultra_sub5m": ResNetSpec(
            name="ultra_sub5m",
            stem_channels=64,
            channels=[64, 144, 208, 280],
            blocks_per_stage=[2, 2, 2, 2],
            conv_kernels=[3, 3, 3, 3],
            skip_kernels=[1, 1, 1, 1],
            pool_size=4,
            dropout_rate=0.0,
        ),
    }
