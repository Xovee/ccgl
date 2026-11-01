"""Contrastive objective and the original CCGL experimental error measures."""

import torch
from torch.nn import functional as F


def contrastive_loss(z, temperature=0.1):
    """NT-Xent: first-view batch followed by its matching second-view batch."""
    z = F.normalize(z, dim=-1)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(-torch.inf)
    targets = (torch.arange(len(z), device=z.device) + len(z) // 2) % len(z)
    return F.cross_entropy(logits, targets)


def training_loss(prediction, target):
    return (prediction.clamp_min(1e-7).log1p()
            - target.clamp_min(1e-7).log1p()).square().mean()


def reported_msle(prediction, target):
    return (prediction.clamp_min(1).log2()
            - target.clamp_min(1).log2()).square().mean()
