"""CIFAR adaptation of LeJEPA's minimal example by Balestriero and LeCun.

Source: https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md
Adaptations: residual CNN, CIFAR-sized views, local logging, FP32 SIGReg.
License: CC BY-NC 4.0; see licenses/lejepa.txt.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T

from model import build_model
from train import MEAN, STD


class SIGReg(nn.Module):
    def __init__(self, directions=256, knots=17):
        super().__init__()
        if directions < 1 or knots < 2:
            raise ValueError('Use positive directions and at least two quadrature knots')
        self.directions = directions
        points = torch.linspace(0, 3, knots)
        step = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * step)
        weights[[0, -1]] = step
        gaussian = torch.exp(-points.square() / 2)
        self.register_buffer('points', points)
        self.register_buffer('gaussian', gaussian)
        self.register_buffer('weights', weights * gaussian)

    def forward(self, projections):
        if projections.ndim != 3 or projections.shape[1] < 2:
            raise ValueError('Expected [views, batch>=2, features]')
        with torch.autocast(projections.device.type, enabled=False):
            projections = projections.float()
            directions = F.normalize(torch.randn(projections.shape[-1], self.directions,
                                                  device=projections.device), dim=0)
            phases = (projections @ directions).unsqueeze(-1) * self.points
            error = (phases.cos().mean(1) - self.gaussian).square() + phases.sin().mean(1).square()
            return ((error @ self.weights) * projections.shape[1]).mean()


class MultiView:
    def __init__(self, views=4):
        if views < 2:
            raise ValueError('LeJEPA requires at least two views')
        self.views = views
        self.transform = T.Compose([
            T.RandomResizedCrop(32, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(), T.Normalize(MEAN, STD),
        ])

    def __call__(self, image):
        return torch.stack([self.transform(image) for _ in range(self.views)])


class LeJEPAEncoder(nn.Module):
    def __init__(self, model_config, hidden=512, projection_dim=64):
        super().__init__()
        self.backbone = build_model(model_config)
        if not hasattr(self.backbone, 'forward_features'):
            raise ValueError('LeJEPA currently supports the WRN residual backbone')
        self.backbone.fc.requires_grad_(False)
        features = self.backbone.fc.in_features
        self.projector = nn.Sequential(
            nn.Linear(features, hidden, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, projection_dim),
        )

    def forward(self, views):
        batch, count = views.shape[:2]
        features = self.backbone.forward_features(views.flatten(0, 1))
        return self.projector(features).reshape(batch, count, -1).transpose(0, 1)


def lejepa_loss(projections, regularizer, weight):
    invariance = (projections.float() - projections.float().mean(0, keepdim=True)).square().mean()
    isotropy = regularizer(projections)
    return (1 - weight) * invariance + weight * isotropy, invariance, isotropy
