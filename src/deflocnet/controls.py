"""Deterministic synthetic user controls; never required at inference.

HED lines and sparse RTV color dots derive from the training target as in the
editing task. Evaluation reports guided editing separately from zero-control
inpainting, since GT-derived guidance contains information about the hole.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw
import torch

from .rtv import RTVConfig, rtv_smooth


class RTVCache:
    def __init__(self, directory: str, existing_directory: str = ""):
        self.directory = Path(directory)
        self.existing = Path(existing_directory) if existing_directory else None

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        digest = hashlib.sha256(rgb.tobytes(order="C")).hexdigest()
        name = Path(digest[:2]) / (digest + ".npy")
        paths = [self.directory / name]
        if self.existing:
            paths.insert(0, self.existing / name)
        for path in paths:
            if not path.exists():
                continue
            try:
                meta = json.loads(path.with_suffix(".json").read_text())
                content = path.read_bytes()
            except FileNotFoundError:
                # Another worker may be committing this deterministic key.
                continue
            if meta["canonical_rgb_sha256"] != digest or hashlib.sha256(content).hexdigest() != meta["structure_sha256"]:
                raise ValueError(f"RTV cache content mismatch: {path}")
            params = meta.get("protocol", {}).get("rtv_parameters")
            if params is not None and params != RTVConfig().as_dict():
                raise ValueError("Incompatible RTV cache parameters")
            arr = np.load(io.BytesIO(content), allow_pickle=False)
            if arr.shape != rgb.shape or arr.dtype != np.float32 or not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
                raise ValueError("Invalid RTV cache RGB")
            return arr
        target = rtv_smooth(rgb).astype(np.float32)
        stream = io.BytesIO()
        np.save(stream, target, allow_pickle=False)
        content = stream.getvalue()
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"canonical_rgb_sha256": digest, "structure_sha256": hashlib.sha256(content).hexdigest(),
                "protocol": {"rtv_parameters": RTVConfig().as_dict(), "rtv_algorithm": "medfe_rtv_matlab_direct_v1"}}
        # Both outputs are deterministic for a content key; replace only complete files.
        for dest, payload in [(path, content), (path.with_suffix(".json"), json.dumps(meta).encode())]:
            tmp = dest.with_name(dest.name + f".{os.getpid()}.tmp")
            tmp.write_bytes(payload)
            tmp.replace(dest)
        return target


def sample_controls(hole: torch.Tensor, rtv: torch.Tensor, rng: random.Random):
    """Equal probabilities: no controls, sketch only, color only, both."""
    _, height, width = hole.shape
    mode = rng.randrange(4)
    dots = Image.new("L", (width, height))
    draw = ImageDraw.Draw(dots)
    if mode in (2, 3):
        ys, xs = torch.where(hole[0])
        for _ in range(rng.randint(1, 12)):
            i = rng.randrange(len(xs))
            x, y = int(xs[i]), int(ys[i])
            radius = rng.randint(2, 6)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    color_mask = torch.from_numpy(np.array(dots, copy=True) > 0).unsqueeze(0) & hole
    color = (2 * rtv - 1) * color_mask
    return {"color": color, "color_mask": color_mask, "sketch_enabled": mode in (1, 3), "control_mode": mode}
