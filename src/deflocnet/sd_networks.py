"""Single-pass RGB U-Net with SD-style residual blocks and spatial controls.

This is an independently trained GAN generator, not a diffusion denoiser or an
SD weight conversion. Sketch and sparse color have separate learned pyramids;
each decoder residual block uses their spatial scale/shift modulation. The
sketch branch gates the color features, retaining the SGB control mechanism.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class FP32GroupNorm(nn.GroupNorm):
    def __init__(self, channels: int, affine: bool = True):
        groups = next(g for g in range(min(32, channels), 0, -1) if channels % g == 0)
        super().__init__(groups, channels, eps=1e-5, affine=affine)

    def forward(self, x: Tensor) -> Tensor:
        if x.dtype not in (torch.float16, torch.bfloat16):
            return super().forward(x)
        with torch.autocast(x.device.type, enabled=False):
            out = F.group_norm(x.float(), self.num_groups,
                               self.weight.float() if self.weight is not None else None,
                               self.bias.float() if self.bias is not None else None, self.eps)
        return out.to(x.dtype)


class ResidualBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__()
        self.norm1, self.norm2 = FP32GroupNorm(incoming), FP32GroupNorm(outgoing)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv2d(incoming, outgoing, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))


class ControlPyramid(nn.Module):
    def __init__(self, incoming: int, channels: list[int]):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(incoming, channels[0], 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(channels[0], channels[0], 3, padding=1), nn.SiLU())
        # Learn local line/dot features BEFORE reduction; do not nearest-sample
        # sparse raw guides, which can remove an entire thin line or color dot.
        self.down = nn.ModuleList(nn.Sequential(
            nn.AvgPool2d(2), nn.Conv2d(a, b, 3, padding=1), nn.SiLU(),
            nn.Conv2d(b, b, 3, padding=1), nn.SiLU()) for a, b in zip(channels, channels[1:]))

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        result = [x]
        for block in self.down:
            x = block(x)
            result.append(x)
        return result


class SpatialControl(nn.Module):
    def __init__(self, guide_channels: int, channels: int):
        super().__init__()
        self.sketch = nn.Conv2d(guide_channels, guide_channels, 3, padding=1)
        self.color = nn.Conv2d(guide_channels, guide_channels, 3, padding=1)
        self.boundary_gate = nn.Conv2d(guide_channels, guide_channels, 3, padding=1)
        self.fuse = nn.Sequential(nn.Conv2d(guide_channels * 2, guide_channels, 3, padding=1), nn.SiLU())
        self.affine = nn.Conv2d(guide_channels, channels * 2, 3, padding=1)
        self.norm = FP32GroupNorm(channels, affine=False)

    def forward(self, h: Tensor, sketch: Tensor, color: Tensor) -> Tensor:
        structure = F.silu(self.sketch(sketch))
        propagated = F.silu(self.color(color)) * (1 - self.boundary_gate(structure).sigmoid())
        gamma, beta = self.affine(self.fuse(torch.cat((structure, propagated), 1))).chunk(2, 1)
        return self.norm(h) * (1 + gamma) + beta


class ControlledResidualBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, guide_channels: int):
        super().__init__()
        self.norm1 = FP32GroupNorm(incoming)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.control = SpatialControl(guide_channels, outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv2d(incoming, outgoing, 1)

    def forward(self, x: Tensor, sketch: Tensor, color: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        return self.skip(x) + self.conv2(F.silu(self.control(h, sketch, color)))


class SpatialAttention(nn.Module):
    """Self-attention only at 16px; PyTorch SDPA avoids a custom attention kernel."""
    def __init__(self, channels: int):
        super().__init__()
        self.heads = next(h for h in range(min(8, channels), 0, -1) if channels % h == 0)
        self.norm = FP32GroupNorm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q, k, v = (p.reshape(b, self.heads, c // self.heads, h * w).transpose(-1, -2)
                   for p in self.qkv(self.norm(x)).chunk(3, 1))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class SDControlGenerator(nn.Module):
    architecture = "sd_controls_v1"
    requires_color_mask = True

    def __init__(self, width: int = 64, activation_checkpoint: bool = True):
        super().__init__()
        self.activation_checkpoint = activation_checkpoint
        channels = [width * m for m in (1, 2, 4, 8, 8)]
        guide_base = max(4, width // 2)
        guide_channels = [guide_base * m for m in (1, 2, 3, 4, 4)]
        # Retain hole noise from the original editor, with ONE explicit validity
        # channel: masked RGB (3), validity (1), hole-restricted noise (3).
        self.stem = nn.Conv2d(7, width, 3, padding=1)
        self.encoder = nn.ModuleList(nn.ModuleList([ResidualBlock(c, c), ResidualBlock(c, c)]) for c in channels)
        self.down = nn.ModuleList(nn.Conv2d(a, b, 3, stride=2, padding=1) for a, b in zip(channels, channels[1:]))
        self.middle = nn.ModuleList([ResidualBlock(channels[-1], channels[-1]),
                                     SpatialAttention(channels[-1]), ResidualBlock(channels[-1], channels[-1])])
        self.sketch_pyramid = ControlPyramid(2, guide_channels)
        self.color_pyramid = ControlPyramid(5, guide_channels)
        self.decoder = nn.ModuleList(nn.ModuleList([
            ControlledResidualBlock(c * 2, c, g), ControlledResidualBlock(c, c, g)
        ]) for c, g in zip(channels, guide_channels))
        self.up = nn.ModuleList(nn.Conv2d(b, a, 3, padding=1) for a, b in zip(channels, channels[1:]))
        self.output = nn.Sequential(FP32GroupNorm(width), nn.SiLU(), nn.Conv2d(width, 3, 3, padding=1), nn.Tanh())
        self.register_buffer("fill", torch.tensor([123., 104., 117.]).view(1, 3, 1, 1) / 127.5 - 1)
        self.apply(self.initialize)
        for module in self.modules():
            if isinstance(module, ControlledResidualBlock):
                # Two 1e-3 projections in series erase condition effects under
                # BF16 rounding. Keep a small, fan-scaled residual with an active
                # modulation path from the first update.
                nn.init.kaiming_normal_(module.conv2.weight, nonlinearity="linear")
                with torch.no_grad():
                    module.conv2.weight.mul_(0.1)
            elif isinstance(module, ResidualBlock):
                nn.init.normal_(module.conv2.weight, std=1e-3)
            elif isinstance(module, SpatialControl):
                nn.init.normal_(module.affine.weight, std=0.02)
            elif isinstance(module, SpatialAttention):
                nn.init.normal_(module.proj.weight, std=1e-3)
        nn.init.normal_(self.output[2].weight, std=0.02)

    @staticmethod
    def initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def run(self, block, *args):
        if self.activation_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    def forward(self, image: Tensor, hole: Tensor, sketch: Tensor, color: Tensor,
                noise: Tensor, *, color_mask: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 32 or any(s % 16 for s in image.shape[-2:]):
            raise ValueError("Expected RGB with image sides >=32 and divisible by 16")
        shape = (image.shape[0], 1, *image.shape[-2:])
        if hole.dtype != torch.bool or hole.shape != shape or color_mask.dtype != torch.bool or color_mask.shape != shape:
            raise ValueError("hole/color_mask must be B1HW boolean tensors")
        if any(t.shape != image.shape for t in (sketch, color, noise)):
            raise ValueError("image/sketch/color/noise must have matching B3HW shapes")
        valid = ~hole
        mask = hole.to(image.dtype)
        color_mask = color_mask & hole
        masked = torch.where(hole, self.fill, image)
        noise = torch.where(hole, noise, torch.zeros_like(noise))
        sketch = torch.where(hole, sketch, torch.zeros_like(sketch)).mean(1, keepdim=True)
        color = torch.where(color_mask, color, torch.zeros_like(color))
        sketches = self.sketch_pyramid(torch.cat((sketch, mask), 1))
        colors = self.color_pyramid(torch.cat((color, color_mask.to(image.dtype), mask), 1))
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
        for i in reversed(range(len(self.decoder))):
            h = torch.cat((h, skips[i]), 1)
            for block in self.decoder[i]:
                h = self.run(block, h, sketches[i], colors[i])
            if i:
                h = self.up[i-1](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.output(h)
