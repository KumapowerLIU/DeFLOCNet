"""Edit one image with an explicit hole and optional user-drawn controls."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from .demo_inference import _load_checkpoint
from .evaluate import pil
from .networks import build_generator, predict_generator
from .runtime import Config, atomic_json


def read_image(path, mode="RGB"):
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert(mode)
        if image.size != (256, 256):
            raise ValueError(f"Expected 256x256 {mode} image: {path}")
        arr = np.array(image, copy=True)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    for arg in ("checkpoint", "image", "mask", "output"):
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--sketch", help="White lines on black background")
    parser.add_argument("--color", help="RGB color hints; requires an explicit color mask")
    parser.add_argument("--color-mask", help="White at supplied color hints, black elsewhere")
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if bool(args.color) != bool(args.color_mask):
        parser.error("--color and --color-mask must be supplied together")
    ck, checksum, _ = _load_checkpoint(args.checkpoint, None)
    if ck["format"] != "deflocnet_modern_v1":
        raise ValueError("Unsupported checkpoint")
    config = Config(**ck["config"])
    model = build_generator(config).to(args.device).eval()
    model.load_state_dict(ck["generator"], strict=True)
    image = read_image(args.image).to(args.device) * 2 - 1
    hole = read_image(args.mask, "L").to(args.device) > .5
    sketch = read_image(args.sketch).to(args.device) if args.sketch else torch.zeros_like(image)
    color = torch.zeros_like(image)
    color_mask = torch.zeros_like(hole)
    if args.color:
        color_mask = read_image(args.color_mask, "L").to(args.device) > .5
        color = (read_image(args.color).to(args.device) * 2 - 1) * color_mask
    rng = torch.Generator(device=args.device).manual_seed(args.seed)
    noise = torch.rand(image.shape, generator=rng, device=args.device)
    raw = predict_generator(model, image, hole, sketch, color, noise, color_mask).float()
    completed = torch.where(hole, raw, image)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pil((completed[0] + 1) / 2).save(output / "completed.png")
    pil((raw[0] + 1) / 2).save(output / "raw.png")
    atomic_json(output / "inference.json", {"step": ck["step"], "checkpoint_sha256": checksum,
                "architecture": config.architecture, "seed": args.seed, "image": args.image, "mask": args.mask,
                "sketch": args.sketch, "color": args.color, "color_mask": args.color_mask})


if __name__ == "__main__":
    main()
