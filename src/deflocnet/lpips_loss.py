"""Official calibrated LPIPS VGG v0.1, loaded strictly from audited local files."""
from __future__ import annotations

from importlib.metadata import version

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .runtime import sha256
from .losses import hole_tv


class FrozenLPIPS(nn.Module):
    def __init__(self, vgg_weights, calibration_weights, vgg_sha256, calibration_sha256):
        super().__init__()
        import lpips
        if version("lpips") != "0.1.4":
            raise ValueError("Use the audited lpips==0.1.4 package")
        for path, expected in ((vgg_weights, vgg_sha256), (calibration_weights, calibration_sha256)):
            if sha256(path) != expected:
                raise ValueError("LPIPS weight checksum mismatch")
        # No implicit downloads and no random weights left in the production metric.
        self.metric = lpips.LPIPS(net="vgg", version="0.1", pretrained=False,
                                  pnet_rand=True, pnet_tune=False, eval_mode=True, verbose=False)
        vgg = torch.load(vgg_weights, map_location="cpu", weights_only=True)
        self.metric.net.load_state_dict({key: vgg["features." + key.split(".", 1)[1]]
                                         for key in self.metric.net.state_dict()}, strict=True)
        calibration = torch.load(calibration_weights, map_location="cpu", weights_only=True)
        expected_keys = {f"lin{i}.model.1.weight" for i in range(5)}
        if set(calibration) != expected_keys:
            raise ValueError("Unexpected VGG LPIPS calibration keys")
        for i in range(5):
            getattr(self.metric, f"lin{i}").load_state_dict(
                {"model.1.weight": calibration[f"lin{i}.model.1.weight"]}, strict=True)
        self.metric.pnet_rand = False
        self.metric.float().eval().requires_grad_(False)
        self.provenance = {"package": "lpips", "package_version": "0.1.4", "net": "vgg", "version": "0.1",
                           "vgg_sha256": vgg_sha256, "calibration_sha256": calibration_sha256,
                           "input_range": [-1, 1], "calibrated": True}
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def _distance(self, prediction, target):
        with torch.autocast(prediction.device.type, enabled=False):
            return self.metric(prediction.float(), target.detach().float(), normalize=False).flatten()

    def forward(self, prediction, target):
        # Chunking bounds FP32 VGG activation memory while preserving per-image averaging.
        values = []
        for p, t in zip(prediction.split(2), target.split(2)):
            distance = checkpoint(self._distance, p, t, use_reentrant=False) if p.requires_grad else self._distance(p, t)
            values.append(distance)
        return torch.cat(values).mean()


def hole_l1(prediction, target, hole):
    error = ((prediction.float() - target.float()).abs() * hole).sum((1, 2, 3))
    count = hole.sum((1, 2, 3)) * prediction.shape[1]
    return (error / count.clamp_min(1)).mean()


class ThreeLossReconstruction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_tv = config.loss_profile in {"l1_lpips_tv_gan_v1", "completion_consistency_v1"}
        if config.loss_profile == "completion_consistency_v1":
            from .consistency import CompletionConsistency
            self.consistency = CompletionConsistency()
            self.use_low_frequency = config.low_frequency_weight > 0
        else:
            self.consistency = None
        self.lpips = FrozenLPIPS(config.vgg_weights, config.lpips_weights,
                                 config.vgg_sha256, config.lpips_sha256)

    def forward(self, fake, target, hole):
        completed = torch.where(hole, fake.float(), target.float())
        parts = {"l1": hole_l1(fake, target, hole), "lpips": self.lpips(completed, target)}
        if self.use_tv:
            parts["tv"] = hole_tv(completed, hole)
        if self.consistency is not None:
            if self.use_low_frequency:
                parts.update(self.consistency(fake, target, hole))
            else:
                from .consistency import boundary_gradient_loss
                parts.update(boundary=boundary_gradient_loss(completed, target, hole),
                             low_frequency=completed.sum() * 0)
        return parts
