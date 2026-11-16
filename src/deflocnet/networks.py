"""Released DeFLOCNet topology, with explicit RGB/control/mask contracts.

The SGB residuals and sigmoid scale match the public implementation. They are
not silently replaced with a different interpretation of the paper equations.
All masks passed to the generator are one-channel booleans: True means hole.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.checkpoint import checkpoint


def norm(channels: int, affine: bool = False) -> nn.Module:
    return nn.InstanceNorm2d(channels, affine=affine, track_running_stats=False)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(2), nn.Conv2d(channels, channels, 3, dilation=2, bias=False),
            norm(channels), nn.ReLU(), nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False), norm(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv_block(x)


class Encoder(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        channels = [9, width, width * 2, width * 4, width * 8, width * 8, width * 8]
        self.blocks = nn.ModuleList()
        for i, (a, b) in enumerate(zip(channels, channels[1:])):
            layers = [] if i == 0 else [nn.LeakyReLU(0.2)]
            layers.append(nn.Conv2d(a, b, 4, 2, 1))
            if 0 < i < 5:
                layers.append(norm(b, affine=True))
            self.blocks.append(nn.Sequential(*layers))
        self.middle = nn.Sequential(*(ResidualBlock(width * 8) for _ in range(4)))

    def forward(self, x: Tensor) -> list[Tensor]:
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        features[-1] = self.middle(features[-1])
        return features


class ControlNorm(nn.Module):
    def __init__(self, condition_channels: int, channels: int, hidden: int, noise_in: bool):
        super().__init__()
        self.noise_in = noise_in
        self.param_free_norm = norm(channels)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(condition_channels * (2 if noise_in else 1), hidden, 3, padding=1), nn.ReLU())
        self.mlp_gamma = nn.Sequential(nn.Conv2d(hidden, channels, 3, padding=1), nn.Sigmoid())
        self.mlp_beta = nn.Conv2d(hidden, channels, 3, padding=1)

    def forward(self, x: Tensor, guide: Tensor, noise: Tensor) -> Tensor:
        if self.noise_in:
            guide = F.interpolate(guide, size=x.shape[-2:], mode="nearest")
            noise = F.interpolate(noise, size=x.shape[-2:], mode="nearest")
            guide = torch.cat((noise, guide), 1)
        features = self.mlp_shared(guide)
        return self.param_free_norm(x) * (1 + self.mlp_gamma(features)) + self.mlp_beta(features)


class SketchBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_0 = ControlNorm(3, 1, 32, noise_in=True)
        self.conv_0 = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x: Tensor, sketch: Tensor, noise: Tensor) -> Tensor:
        return x + self.conv_0(F.leaky_relu(self.norm_0(x, sketch, noise), 0.2))


class ColorBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gated_se = nn.Sequential(
            norm(1), nn.ReLU(), nn.Conv2d(1, channels, 3, padding=1),
            norm(channels), nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1), nn.Sigmoid())
        self.convcolor_f = nn.Conv2d(channels, channels, 3, padding=1)
        self.normcolor_f = norm(channels)
        self.accolor_f = nn.LeakyReLU(0.2)
        self.convcolor = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1))
        self.color_to_f = nn.Conv2d(3, channels, 1)
        # The original uses slope 0.01 in this subbranch, unlike other SGB activations.
        self.convn = nn.Sequential(norm(channels), nn.LeakyReLU(0.01), nn.Conv2d(channels, channels, 3, padding=1))
        self.norm = ControlNorm(channels, channels, channels // 2, noise_in=False)

    def forward(self, x: Tensor, sketch: Tensor, color: Tensor, noise: Tensor, hole: Tensor):
        if color.shape[1] == 3:
            color = self.color_to_f(F.interpolate(color, size=x.shape[-2:], mode="nearest"))
        gate = self.gated_se(sketch)
        fused = x + self.convn(x) * gate
        mask = F.interpolate(hole, size=x.shape[-2:], mode="nearest")
        color = self.accolor_f(self.normcolor_f((1 - gate) * self.convcolor_f(color * mask)))
        update = self.convcolor(F.leaky_relu(self.norm(fused, color, noise), 0.2))
        return x + (fused + update), color


class StructureGenerationBlock(nn.Module):
    def __init__(self, channels: int, depth: int):
        super().__init__()
        self.sk_blocks = nn.ModuleList(SketchBlock() for _ in range(depth))
        self.co_bolcks = nn.ModuleList(ColorBlock(channels) for _ in range(depth))

    def forward(self, x: Tensor, sketch: Tensor, color: Tensor, noise: Tensor, hole: Tensor) -> Tensor:
        lines, fused = x.mean(1, keepdim=True), x
        for sk, co in zip(self.sk_blocks, self.co_bolcks):
            lines = sk(lines, sketch, noise)
            fused, color = co(fused, lines, color, noise, hole)
        return fused + x


def up_block(input_channels: int, output_channels: int, last: bool = False) -> nn.Sequential:
    return nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(input_channels, output_channels, 4, 2, 1),
                         nn.Tanh() if last else norm(output_channels, affine=True))


class Decoder(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        structure = [(8, 8), (16, 8), (16, 4), (8, 2), (4, 1)]
        texture = [(8, 8), (8, 8), (8, 4), (4, 2), (2, 1)]
        self.structure = nn.ModuleList(up_block(a * width, b * width) for a, b in structure)
        self.texture = nn.ModuleList(up_block(a * width, b * width) for a, b in texture)
        self.output = up_block(width * 2, 3, last=True)

    def forward(self, skips: list[Tensor], bottleneck: Tensor) -> Tensor:
        texture, x = bottleneck, skips[-1]
        for i, (s, t) in enumerate(zip(self.structure, self.texture)):
            texture = t(texture)
            if i:
                x = torch.cat((x, skips[-1 - i]), 1)
            x = s(x) + texture
        return self.output(torch.cat((x, skips[0]), 1))


class Generator(nn.Module):
    def __init__(self, width: int = 64, checkpoint_sgb: bool = True):
        super().__init__()
        self.width, self.checkpoint_sgb = width, checkpoint_sgb
        self.encoder = Encoder(width)
        self.sgb = nn.ModuleList(StructureGenerationBlock(c * width, d)
                                 for c, d in zip((1, 2, 4, 8, 8, 8), (6, 5, 4, 3, 2, 1)))
        self.decoder = Decoder(width)
        self.register_buffer("fill", torch.tensor([123., 104., 117.]).view(1, 3, 1, 1) / 127.5 - 1)
        self.apply(initialize)

    def forward(self, image: Tensor, hole: Tensor, sketch: Tensor, color: Tensor, noise: Tensor) -> Tensor:
        if hole.dtype != torch.bool or hole.shape != (image.shape[0], 1, *image.shape[-2:]):
            raise ValueError("hole must be B1HW boolean, True means missing")
        if image.shape[1] != 3 or any(t.shape != image.shape for t in (sketch, color, noise)):
            raise ValueError("RGB image/sketch/color/noise must have identical B3HW shape")
        if image.shape[-2] % 64 or image.shape[-1] % 64 or min(image.shape[-2:]) < 192:
            raise ValueError("Image sides must be multiples of 64 and at least 192")
        masked = torch.where(hole, self.fill, image)
        mask = hole.to(image.dtype)
        noise, sketch, color = noise * mask, sketch * mask, color * mask
        encoded = self.encoder(torch.cat((masked, (~hole).expand_as(image).to(image.dtype), noise), 1))
        skips = []
        for block, x in zip(self.sgb, encoded):
            args = (x, sketch, color, noise, mask)
            if self.checkpoint_sgb and self.training and torch.is_grad_enabled():
                skips.append(checkpoint(block, *args, use_reentrant=False))
            else:
                skips.append(block(*args))
        return self.decoder(skips, encoded[-1])


def initialize(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels: int, width: int = 64, layers: int = 4):
        super().__init__()
        blocks = [nn.Sequential(nn.Conv2d(input_channels, width, 4, 2, 2), nn.LeakyReLU(0.2))]
        channels = width
        for i in range(1, layers):
            out = min(width * 2 ** i, 512)
            conv = nn.Conv2d(channels, out, 4, 1 if i == layers - 1 else 2, 2, bias=False)
            initialize(conv)
            blocks.append(nn.Sequential(spectral_norm(conv), norm(out), nn.LeakyReLU(0.2)))
            channels = out
        blocks.append(nn.Sequential(nn.Conv2d(channels, 1, 4, 1, 2)))
        blocks[0].apply(initialize)
        blocks[-1].apply(initialize)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_channels: int, width: int = 64, scales: int = 2):
        super().__init__()
        self.scales = nn.ModuleList(PatchDiscriminator(input_channels, width) for _ in range(scales))

    def forward(self, x: Tensor) -> list[Tensor]:
        result = []
        for i, discriminator in enumerate(self.scales):
            if i:
                x = F.avg_pool2d(x, 3, 2, 1, count_include_pad=False)
            result.append(discriminator(x))
        return result


class Discriminators(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.global_d = MultiScaleDiscriminator(6, width)
        self.local_d = MultiScaleDiscriminator(3, width)

    def forward(self, image: Tensor, hole: Tensor, crop: tuple[int, int]) -> list[list[Tensor]]:
        y, x = crop
        return [self.global_d(torch.cat((hole.expand_as(image).to(image.dtype), image), 1)),
                self.local_d(image[:, :, y:y + 64, x:x + 64])]


def build_generator(config):
    if config.architecture == "sd_color_v2":
        from .sd_color_networks import SDColorGenerator
        return SDColorGenerator(config.width, config.checkpoint_sgb)
    if config.architecture == "sd_controls_v1":
        from .sd_networks import SDControlGenerator
        return SDControlGenerator(config.width, config.checkpoint_sgb)
    if config.architecture == "released_v1":
        return Generator(config.width, config.checkpoint_sgb)
    raise ValueError(f"Unknown generator architecture: {config.architecture}")


def predict_generator(generator, image, hole, sketch, color, noise, color_mask=None):
    if getattr(generator, "requires_color_mask", False):
        if color_mask is None:
            raise ValueError("This generator requires an explicit color_mask")
        return generator(image, hole, sketch, color, noise, color_mask=color_mask)
    return generator(image, hole, sketch, color, noise)
