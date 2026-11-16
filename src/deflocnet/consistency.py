"""Supervise compositing seams and low-frequency RGB without blurring GT edges."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def boundary_gradient_loss(completed, target, hole, radius=4):
    """Match GT horizontal/vertical gradients near the mask boundary.

    A pair participates when it touches a hole and the 4px band around a
    hole/known transition. Actual GT gradients are preserved, including edges.
    Each image is normalized by its number of participating RGB pairs.
    """
    with torch.autocast(completed.device.type, enabled=False):
        residual = completed.float() - target.detach().float()
        h = hole.float()
        band = (F.max_pool2d(h, 2 * radius + 1, 1, radius) > 0) & (
            F.max_pool2d(1 - h, 2 * radius + 1, 1, radius) > 0
        )
        xmask = (hole[..., 1:] | hole[..., :-1]) & (band[..., 1:] | band[..., :-1])
        ymask = (hole[..., 1:, :] | hole[..., :-1, :]) & (band[..., 1:, :] | band[..., :-1, :])
        dx = (residual[..., 1:] - residual[..., :-1]).abs()
        dy = (residual[..., 1:, :] - residual[..., :-1, :]).abs()
        numerator = (dx * xmask).sum((1, 2, 3)) + (dy * ymask).sum((1, 2, 3))
        count = (xmask.sum((1, 2, 3)) + ymask.sum((1, 2, 3))) * completed.shape[1]
        return (numerator / count.clamp_min(1)).mean()


class CompletionConsistency(nn.Module):
    """Gaussian-filtered error detects broad color shifts at two spatial scales."""

    def __init__(self):
        super().__init__()
        for i, (radius, sigma) in enumerate(((7, 3.0), (15, 7.0))):
            x = torch.arange(-radius, radius + 1, dtype=torch.float32)
            kernel = torch.exp(-x.square() / (2 * sigma * sigma))
            self.register_buffer(f"kernel_{i}", kernel / kernel.sum())

    @staticmethod
    def blur(x, kernel):
        radius = kernel.numel() // 2
        channels = x.shape[1]
        horizontal = kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1)
        vertical = kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1)
        x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), horizontal, groups=channels)
        return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), vertical, groups=channels)

    def forward(self, fake, target, hole):
        with torch.autocast(fake.device.type, enabled=False):
            # Known pixels and the target never receive reconstruction gradients.
            target = target.detach().float()
            completed = torch.where(hole, fake.float(), target)
            residual = completed - target
            packed = torch.cat((residual, hole.float()), 1)
            values = []
            for i in range(2):
                filtered = self.blur(packed, getattr(self, f"kernel_{i}"))
                # Normalize by blurred hole mass; do not dilute small holes with
                # unchanged known pixels. Filter ERROR, not output texture.
                numerator = filtered[:, :3].abs().sum((1, 2, 3))
                denominator = filtered[:, 3:].sum((1, 2, 3)) * 3
                values.append(numerator / denominator.clamp_min(1))
            return {"boundary": boundary_gradient_loss(completed, target, hole),
                    "low_frequency": torch.stack(values).mean()}
