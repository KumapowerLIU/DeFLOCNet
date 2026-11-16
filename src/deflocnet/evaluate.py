"""Small fixed evaluations: guided editing and unguided inpainting separately."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .data import IndexedImages
from .edges import hed_sketch
from .hed import HED
from .networks import build_generator, predict_generator
from .runtime import Config, atomic_json, sha256


def make_dataset(config: Config, training: bool):
    return IndexedImages(config.train_manifest if training else config.validation_manifest,
                         size=config.image_size, seed=config.seed, training=training,
                         expected_sha256=config.train_sha256 if training else config.validation_sha256,
                         min_fraction=config.mask_min_fraction, max_fraction=config.mask_max_fraction,
                         rtv_cache=config.rtv_cache, existing_rtv_cache=config.existing_rtv_cache)


def pil(tensor):
    arr = tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint(arr * 255).astype(np.uint8))


@torch.inference_mode()
def evaluate(generator, hed, data, output, step: int, indices=None, *, hed_postprocess="threshold_v1"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = next(generator.parameters()).device
    was_training = generator.training
    generator.eval()
    indices = indices or [0, 133, 533, 1599]
    rows, metrics = [], []
    for index in indices:
        item = data[index]
        rgb = item["image"].unsqueeze(0).to(device)
        image = rgb * 2 - 1
        hole = (~item["valid"]).unsqueeze(0).to(device)
        rng = torch.Generator(device=device).manual_seed(20261101 + index)
        noise = torch.rand(image.shape, generator=rng, device=device)
        sketch = hed_sketch(hed, rgb, hed_postprocess) * hole
        color = item["color"].unsqueeze(0).to(device)
        color_mask = item["color_mask"].unsqueeze(0).to(device)
        # Explicit guided test: all HED lines; sparse RTV color if supplied.
        guided = predict_generator(generator, image, hole, sketch, color, noise, color_mask)
        unguided = predict_generator(generator, image, hole, torch.zeros_like(image), torch.zeros_like(image), noise,
                                       torch.zeros_like(color_mask))
        panels = [torch.where(hole, (generator.fill + 1) / 2, rgb), sketch,
                  torch.where(hole, (guided.float() + 1) / 2, rgb),
                  torch.where(hole, (unguided.float() + 1) / 2, rgb), rgb]
        row = Image.new("RGB", (256 * 5, 280), "white")
        draw = ImageDraw.Draw(row)
        for i, (title, panel) in enumerate(zip(("Masked", "HED guide", "Guided", "No controls", "GT"), panels)):
            row.paste(pil(panel[0]), (256 * i, 24))
            draw.text((256 * i + 6, 6), f"{index}: {title}", fill="black")
        row.save(output / f"case-{index:06d}.png")
        row.save(output / f"case-{index:06d}.jpg", quality=95)
        rows.append(row)
        for name, prediction in (("guided", guided), ("unguided", unguided)):
            error = ((prediction.float() + 1) / 2 - rgb)
            mse = (error.square() * hole).sum() / (hole.sum() * 3)
            l1 = (error.abs() * hole).sum() / (hole.sum() * 3)
            metrics.append({"index": index, "mode": name, "hole_psnr": float(-10 * mse.clamp_min(1e-12).log10()), "hole_l1": float(l1)})
    grid = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    for i, row in enumerate(rows):
        grid.paste(row, (0, i * row.height))
    grid.save(output / "preview.jpg", quality=95)
    grid.save(output / "preview.png")
    atomic_json(output / "metrics.json", {"step": step, "rows": metrics, "hed_postprocess": hed_postprocess,
                "note": "GT-derived HED/RTV guidance is not an unguided inpainting benchmark."})
    generator.train(was_training)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = Config(**ck["config"])
    config.verify_weights()
    generator = build_generator(config).to(args.device)
    generator.load_state_dict(ck["generator"], strict=True)
    hed = HED(config.hed_weights).to(args.device)
    data = make_dataset(config, False)
    evaluate(generator, hed, data, args.output, ck["step"], hed_postprocess=config.hed_postprocess)
    atomic_json(Path(args.output) / "checkpoint.json", {"path": str(Path(args.checkpoint).resolve()),
                "sha256": sha256(args.checkpoint), "step": ck["step"]})


if __name__ == "__main__":
    main()
