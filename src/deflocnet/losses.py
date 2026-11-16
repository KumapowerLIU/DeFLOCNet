"""Five reference loss families; corrected VGG preprocessing and TV indexing."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.vgg import make_layers, cfgs
from torch.utils.checkpoint import checkpoint


def hinge_discriminator(real, fake) -> Tensor:
    # Average scales, SUM global/local, exactly as released GANLoss + reduce.
    return sum(sum((F.relu(1 - r.float()).mean() + F.relu(1 + f.float()).mean())
                   for r, f in zip(rs, fs)) / len(rs) for rs, fs in zip(real, fake))


def hinge_generator(fake) -> Tensor:
    return sum(sum(-f.float().mean() for f in scale) / len(scale) for scale in fake)


def hole_tv(image: Tensor, hole: Tensor) -> Tensor:
    # Paper Eq.10: adjacent pairs inside hole, normalized by full image elements.
    image = image.float()
    horizontal = hole[:, :, :, 1:] & hole[:, :, :, :-1]
    vertical = hole[:, :, 1:, :] & hole[:, :, :-1, :]
    return (((image[:, :, :, 1:] - image[:, :, :, :-1]).abs() * horizontal).sum()
            + ((image[:, :, 1:, :] - image[:, :, :-1, :]).abs() * vertical).sum()) / image.numel()


def gram(x: Tensor) -> Tensor:
    x = x.float()
    batch, channels, height, width = x.shape
    features = x.reshape(batch, channels, -1)
    return features @ features.transpose(1, 2) / (channels * height * width)


class VGGFeatures(nn.Module):
    perceptual_indices = (1, 6, 11, 18, 25)
    style_indices = (8, 15, 22, 27)

    def __init__(self, weights: str):
        super().__init__()
        self.features = make_layers(cfgs["D"])[:28]
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.features.load_state_dict({key.removeprefix("features."): value for key, value in state.items()
                                       if key.startswith("features.") and int(key.split(".")[1]) < 28}, strict=True)
        for module in self.features.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        self.requires_grad_(False).eval()
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> dict[int, Tensor]:
        x = ((x.float() + 1) / 2 - self.mean) / self.std
        out, begin = {}, 0
        for end in sorted(set(self.perceptual_indices + self.style_indices)):
            block = self.features[begin:end + 1]
            x = checkpoint(block, x, use_reentrant=False) if x.requires_grad else block(x)
            out[end] = x
            begin = end + 1
        return out


class ReconstructionLoss(nn.Module):
    def __init__(self, vgg_weights: str):
        super().__init__()
        self.vgg = VGGFeatures(vgg_weights)

    def forward(self, fake: Tensor, target: Tensor, hole: Tensor) -> dict[str, Tensor]:
        fake, target = fake.float(), target.float()
        completed = torch.where(hole, fake, target)
        # Public training uses both raw and completed images for VGG/style.
        with torch.autocast(device_type=fake.device.type, enabled=False):
            with torch.no_grad():
                ground = self.vgg(target)
            perceptual = fake.new_zeros(())
            style = fake.new_zeros(())
            for image in (fake, completed):
                features = self.vgg(image)
                perceptual = perceptual + sum(F.l1_loss(features[i], ground[i]) for i in self.vgg.perceptual_indices)
                style = style + sum(F.l1_loss(gram(features[i]), gram(ground[i])) for i in self.vgg.style_indices)
        # Public code's known-region + 2 * hole L1, both averaged over B*3*H*W.
        return {"l1": ((fake - target).abs() * (1 + hole.to(fake.dtype))).mean(),
                "perceptual": perceptual, "style": style, "tv": hole_tv(completed, hole)}
