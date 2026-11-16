"""SD RGB backbone with normalized, recurrent color propagation.

Version 2 keeps sparse RGB values separate from their sampling density, carries
color state through successive decoder blocks, and gives color its own affine
head. This is an architectural experiment, not an exact original-SGB replica.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .sd_networks import FP32GroupNorm, SDControlGenerator


def pooled_color_guides(color: Tensor, valid: Tensor, hole: Tensor, size: tuple[int, int]) -> Tensor:
    """Mean supplied RGB per cell; empty pixels never dilute a supplied color.

    RGB stays in [-1,1]. Presence distinguishes a genuine gray hint (RGB=0)
    from an empty cell. Pool sums/counts in FP32, including under BF16 AMP.
    """
    valid = valid & hole
    with torch.autocast(color.device.type, enabled=False):
        weight = valid.float()
        density = F.adaptive_avg_pool2d(weight, size)
        numerator = F.adaptive_avg_pool2d(torch.where(valid, color.float(), 0.0), size)
        present = density > 0
        rgb = torch.where(present, numerator / density.clamp_min(1e-12), 0.0)
        area = F.adaptive_avg_pool2d(hole.float(), size)
        return torch.cat((rgb, present.float(), area), dim=1)


class SparseColorPyramid(nn.Module):
    def __init__(self, channels: list[int]):
        super().__init__()
        self.levels = nn.ModuleList(nn.Sequential(
            nn.Conv2d(5, c, 3, padding=1), FP32GroupNorm(c), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1), FP32GroupNorm(c), nn.SiLU(),
        ) for c in channels)

    def forward(self, color: Tensor, valid: Tensor, hole: Tensor) -> list[Tensor]:
        enabled = (valid & hole).flatten(1).any(1).view(-1, 1, 1, 1)
        height, width = color.shape[-2:]
        result = []
        for i, level in enumerate(self.levels):
            guides = pooled_color_guides(color, valid, hole, (height // 2**i, width // 2**i))
            feature = level(guides)
            result.append(torch.where(enabled, feature, torch.zeros_like(feature)))
        return result


class PropagatingColorControl(nn.Module):
    def __init__(self, guide_channels: int, channels: int):
        super().__init__()
        self.sketch = nn.Conv2d(guide_channels, guide_channels, 3, padding=1)
        self.boundary_gate = nn.Conv2d(guide_channels, guide_channels, 3, padding=1)
        self.color = nn.Conv2d(guide_channels, guide_channels, 3, padding=1)
        self.color_norm = FP32GroupNorm(guide_channels)
        self.sketch_affine = nn.Conv2d(guide_channels, channels * 2, 3, padding=1)
        self.color_affine = nn.Conv2d(guide_channels, channels * 2, 3, padding=1)
        self.norm = FP32GroupNorm(channels, affine=False)

    def forward(self, h: Tensor, sketch: Tensor, color: Tensor, enabled: Tensor) -> tuple[Tensor, Tensor]:
        structure = F.silu(self.sketch(sketch))
        propagated = self.color(color) * (1 - self.boundary_gate(structure).sigmoid())
        # Restore the original SGB ordering: gate -> feature norm -> activation.
        # Normalize learned features, never the supplied RGB itself.
        propagated = F.leaky_relu(self.color_norm(propagated), negative_slope=0.2)
        propagated = torch.where(enabled, propagated, torch.zeros_like(propagated))
        color_affine = self.color_affine(propagated)
        color_affine = torch.where(enabled, color_affine, torch.zeros_like(color_affine))
        gamma, beta = (self.sketch_affine(structure) + color_affine).chunk(2, 1)
        return self.norm(h) * (1 + gamma) + beta, propagated


class ColorResidualBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, guide_channels: int):
        super().__init__()
        self.norm1 = FP32GroupNorm(incoming)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.control = PropagatingColorControl(guide_channels, outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv2d(incoming, outgoing, 1)

    def forward(self, x: Tensor, sketch: Tensor, color: Tensor, enabled: Tensor) -> tuple[Tensor, Tensor]:
        h = self.conv1(F.silu(self.norm1(x)))
        h, color = self.control(h, sketch, color, enabled)
        return self.skip(x) + self.conv2(F.silu(h)), color


class SDColorGenerator(SDControlGenerator):
    architecture = 'sd_color_v2'

    def __init__(self, width: int = 64, activation_checkpoint: bool = True):
        super().__init__(width, activation_checkpoint)
        channels = [width * m for m in (1, 2, 4, 8, 8)]
        guide_channels = [max(4, width // 2) * m for m in (1, 2, 3, 4, 4)]
        self.color_pyramid = SparseColorPyramid(guide_channels)
        self.decoder = nn.ModuleList(nn.ModuleList([
            ColorResidualBlock(c * 2, c, g), ColorResidualBlock(c, c, g)
        ]) for c, g in zip(channels, guide_channels))
        self.color_up = nn.ModuleList(nn.Sequential(
            nn.Conv2d(b, a, 1), FP32GroupNorm(a), nn.SiLU()
        ) for a, b in zip(guide_channels, guide_channels[1:]))
        for module in (self.color_pyramid, self.decoder, self.color_up):
            module.apply(self.initialize)
        for module in self.decoder.modules():
            if isinstance(module, ColorResidualBlock):
                nn.init.kaiming_normal_(module.conv2.weight, nonlinearity='linear')
                with torch.no_grad():
                    module.conv2.weight.mul_(0.1)
            elif isinstance(module, PropagatingColorControl):
                nn.init.normal_(module.sketch_affine.weight, std=0.02)
                nn.init.normal_(module.color_affine.weight, std=0.02)

    def forward(self, image: Tensor, hole: Tensor, sketch: Tensor, color: Tensor,
                noise: Tensor, *, color_mask: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 32 or any(s % 16 for s in image.shape[-2:]):
            raise ValueError('Expected RGB with image sides >=32 and divisible by 16')
        shape = (image.shape[0], 1, *image.shape[-2:])
        if hole.dtype != torch.bool or hole.shape != shape or color_mask.dtype != torch.bool or color_mask.shape != shape:
            raise ValueError('hole/color_mask must be B1HW boolean tensors')
        if any(t.shape != image.shape for t in (sketch, color, noise)):
            raise ValueError('image/sketch/color/noise must have matching B3HW shapes')
        valid = ~hole
        color_mask = color_mask & hole
        enabled = color_mask.flatten(1).any(1).view(-1, 1, 1, 1)
        masked = torch.where(hole, self.fill, image)
        noise = torch.where(hole, noise, torch.zeros_like(noise))
        sketch = torch.where(hole, sketch, torch.zeros_like(sketch)).mean(1, keepdim=True)
        color = torch.where(color_mask, color, torch.zeros_like(color))
        sketches = self.sketch_pyramid(torch.cat((sketch, hole.to(image.dtype)), 1))
        colors = self.color_pyramid(color, color_mask, hole)
        h = self.stem(torch.cat((masked, valid.to(image.dtype), noise), 1))
        skips = []
        for i, blocks in enumerate(self.encoder):
            for block in blocks:
                h = self.run(block, h)
            skips.append(h)
            if i < len(self.down):
                h = self.down[i](h)
        for block in self.middle:
            h = self.run(block, h)
        state = None
        for i in reversed(range(len(self.decoder))):
            if state is None:
                state = colors[i]
            else:
                carried = self.color_up[i](F.interpolate(state, size=colors[i].shape[-2:], mode='nearest'))
                state = (colors[i] + carried) / math.sqrt(2)
                state = torch.where(enabled, state, torch.zeros_like(state))
            h = torch.cat((h, skips[i]), 1)
            for block in self.decoder[i]:
                h, state = self.run(block, h, sketches[i], state, enabled)
            if i:
                h = self.up[i-1](F.interpolate(h, scale_factor=2, mode='nearest'))
        return self.output(h)
