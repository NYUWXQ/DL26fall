"""Single-GPU VICReg auxiliary loss, adapted from the authors' implementation.

Source: facebookresearch/vicreg/main_vicreg.py (MIT; licenses/vicreg.txt).
The projector is used during training and omitted from classifier exports.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ThreeViews:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        return tuple(self.transform(image) for _ in range(3))


def projector(features):
    return nn.Sequential(nn.Linear(features, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                         nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                         nn.Linear(1024, 1024, bias=False))


def vicreg_loss(first, second):
    if first.ndim != 2 or first.shape != second.shape or first.shape[0] < 2:
        raise ValueError('VICReg requires matching [batch>=2, features] projections')
    with torch.autocast(first.device.type, enabled=False):
        first, second = first.float(), second.float()
        invariance = F.mse_loss(first, second)
        centered = [value - value.mean(0) for value in (first, second)]
        variance = sum(F.relu(1 - (value.var(0) + 1e-4).sqrt()).mean()
                       for value in centered) / 2
        covariance = first.new_zeros(())
        for value in centered:
            cov = value.T @ value / (len(value) - 1)
            off_diagonal = cov.flatten()[:-1].view(cov.shape[0] - 1, cov.shape[0] + 1)[:, 1:]
            covariance = covariance + off_diagonal.square().sum() / cov.shape[0]
        return 25 * invariance + 25 * variance + covariance


def auxiliary_loss(features, head, views):
    # Separate forwards retain the supervised batch's normalization statistics.
    return vicreg_loss(head(features(views[0])), head(features(views[1])))
