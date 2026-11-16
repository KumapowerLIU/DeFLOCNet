"""Atomic artifacts, strict source/config provenance, and restartable cursors."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def source_hashes() -> dict:
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


@dataclass(frozen=True)
class Config:
    train_manifest: str
    validation_manifest: str
    train_sha256: str
    validation_sha256: str
    vgg_weights: str
    hed_weights: str
    vgg_sha256: str
    hed_sha256: str
    rtv_cache: str
    existing_rtv_cache: str = ""
    seed: int = 2019
    image_size: int = 256
    width: int = 64
    discriminator_width: int = 64
    batch_size: int = 64
    microbatch_size: int = 2
    workers: int = 16
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    l1_weight: float = 1.0
    perceptual_weight: float = 0.05
    style_weight: float = 250.0
    tv_weight: float = 0.1
    gan_weight: float = 0.1
    max_steps: int = 100000
    save_every: int = 500
    evaluate_every: int = 500
    keep_checkpoints: int = 2
    memory_budget_gib: float = 15.0
    amp: str = "bf16"
    checkpoint_sgb: bool = True
    mask_min_fraction: float = 0.1
    mask_max_fraction: float = 0.4
    # Missing in legacy checkpoints: preserve their original guide semantics.
    hed_postprocess: str = "threshold_v1"
    architecture: str = "released_v1"
    loss_profile: str = "released_five_v1"
    lpips_weight: float = 0.0
    lpips_weights: str = ""
    lpips_sha256: str = ""
    boundary_weight: float = 0.0
    low_frequency_weight: float = 0.0

    def __post_init__(self):
        for key in ("width", "discriminator_width", "batch_size", "microbatch_size", "max_steps", "save_every", "evaluate_every", "keep_checkpoints"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"Invalid {key}")
        if self.batch_size % self.microbatch_size or self.workers < 0:
            raise ValueError("Invalid batch/microbatch/workers")
        if self.image_size != 256 or self.width < 4:
            raise ValueError("This reference training profile requires 256px and width >=4")
        if self.amp not in {"bf16", "fp32"}:
            raise ValueError("Only BF16 and FP32 are supported")
        if self.hed_postprocess not in {"threshold_v1", "hed_nms_thin_v1", "hed_nms_medium_v1"}:
            raise ValueError("Unknown HED guide protocol")
        if self.architecture not in {"released_v1", "sd_controls_v1", "sd_color_v2"}:
            raise ValueError("Unknown generator architecture")
        if self.loss_profile not in {"released_five_v1", "l1_lpips_gan_v1", "l1_lpips_tv_gan_v1", "completion_consistency_v1"}:
            raise ValueError("Unknown loss profile")
        for key in ("l1_weight", "perceptual_weight", "style_weight", "tv_weight", "gan_weight", "lpips_weight", "learning_rate", "memory_budget_gib", "boundary_weight", "low_frequency_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"Invalid {key}")
        if self.loss_profile != "completion_consistency_v1" and (self.boundary_weight or self.low_frequency_weight):
            raise ValueError("Consistency losses require the explicit consistency profile")
        if self.loss_profile == "completion_consistency_v1" and self.boundary_weight <= 0:
            raise ValueError("Consistency profile requires positive boundary weight")
        if self.loss_profile in {"l1_lpips_gan_v1", "l1_lpips_tv_gan_v1", "completion_consistency_v1"}:
            disabled = ("perceptual_weight", "style_weight")
            if self.loss_profile == "l1_lpips_gan_v1":
                disabled += ("tv_weight",)
            if any(getattr(self, k) != 0 for k in disabled):
                raise ValueError("LPIPS profile must disable " + "/".join(disabled))
            if self.loss_profile in {"l1_lpips_tv_gan_v1", "completion_consistency_v1"} and self.tv_weight <= 0:
                raise ValueError("LPIPS plus TV profile requires positive TV weight")
            if min(self.l1_weight, self.lpips_weight, self.gan_weight) <= 0 or not self.lpips_weights or len(self.lpips_sha256) != 64:
                raise ValueError("LPIPS profile requires positive L1/LPIPS/GAN and pinned LPIPS weights")
        elif self.lpips_weight:
            raise ValueError("LPIPS requires an LPIPS loss profile")
        if not 0 < self.mask_min_fraction < self.mask_max_fraction < 1:
            raise ValueError("Invalid mask fractions")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def as_dict(self):
        return asdict(self)

    def verify_weights(self):
        for name in (("vgg", "hed", "lpips") if self.loss_profile in {"l1_lpips_gan_v1", "l1_lpips_tv_gan_v1", "completion_consistency_v1"} else ("vgg", "hed")):
            if sha256(getattr(self, name + "_weights")) != getattr(self, name + "_sha256"):
                raise ValueError(f"{name} weights differ from config")


def capture_rng():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(), "numpy": np.random.get_state()}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def save_checkpoint(output, step, generator, discriminator, optim_g, optim_d, config, epoch, cursor, lineage=None):
    output = Path(output)
    payload = {"format": "deflocnet_modern_v1", "step": step, "epoch": epoch, "cursor": cursor,
               "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
               "optimizer_g": optim_g.state_dict(), "optimizer_d": optim_d.state_dict(),
               "config": config.as_dict(), "source": source_hashes(), "rng": capture_rng(), "saved_at": time.time()}
    if lineage is not None:
        payload["lineage"] = lineage
    path = output / f"checkpoint-{step:08d}.pt"
    temp = path.with_suffix(".pt.tmp")
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    link = output / f"latest.{os.getpid()}.tmp"
    link.symlink_to(path.name)
    link.replace(output / "latest.pt")
    old = sorted(output.glob("checkpoint-*.pt"))[:-config.keep_checkpoints]
    for stale in old:
        stale.unlink()
    return path
