"""CSA random-mask sampler, reused without changing its distribution.

Source: KumapowerLIU/CSA-inpainting, src/csa_inpainting/masks.py.
Output: True means hole (converted to valid masks by the dataset).
"""

import math
import random

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor


def random_mask(
    height: int,
    width: int,
    rng: random.Random,
    min_fraction: float = 0.1,
    max_fraction: float = 0.4,
) -> Tensor:
    """Sample rectangles, free-form strokes, or their union at random locations.

    The supplied RNG makes masks independent of global/worker RNG state. Rejection
    sampling bounds the missing area; this is a synthetic training distribution,
    not the irregular-mask benchmark released with the original paper.
    """
    if height < 2 or width < 2:
        raise ValueError("Random mask dimensions must be at least 2")
    if not 0 < min_fraction < max_fraction < 1:
        raise ValueError("Require 0 < min_fraction < max_fraction < 1")
    pixels = height * width
    minimum, maximum = math.ceil(pixels * min_fraction), math.floor(pixels * max_fraction)
    if minimum > maximum:
        raise ValueError("Mask fraction range contains no whole-pixel area")
    short_side = min(height, width)
    for _ in range(32):
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        kind = rng.choice(("rectangles", "strokes", "mixed"))
        if kind in {"rectangles", "mixed"}:
            for _ in range(rng.randint(1, 3)):
                box_h = rng.randint(max(1, height // 8), max(1, height * 2 // 3))
                box_w = rng.randint(max(1, width // 8), max(1, width * 2 // 3))
                top, left = rng.randrange(height - box_h + 1), rng.randrange(width - box_w + 1)
                draw.rectangle((left, top, left + box_w - 1, top + box_h - 1), fill=255)
        if kind in {"strokes", "mixed"}:
            for _ in range(rng.randint(2, 6)):
                x, y = rng.randrange(width), rng.randrange(height)
                for _ in range(rng.randint(1, 4)):
                    angle = rng.uniform(0, 2 * math.pi)
                    length = rng.uniform(short_side / 16, short_side / 2)
                    nx = max(0, min(width - 1, round(x + length * math.cos(angle))))
                    ny = max(0, min(height - 1, round(y + length * math.sin(angle))))
                    stroke = rng.randint(max(1, short_side // 25), max(2, short_side // 7))
                    draw.line((x, y, nx, ny), fill=255, width=stroke)
                    radius = stroke // 2
                    for cx, cy in ((x, y), (nx, ny)):
                        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
                    x, y = nx, ny
        array = np.array(canvas, copy=True) > 0
        if minimum <= int(array.sum()) <= maximum:
            return torch.from_numpy(array).unsqueeze(0)

    # Very narrow/custom bounds may defeat rejection sampling. Use a randomly
    # placed rectangle with a partial final row to guarantee an admissible area.
    count = rng.randint(minimum, maximum)
    box_w = rng.randint(math.ceil(count / height), min(width, count))
    full_rows, remainder = divmod(count, box_w)
    box_h = full_rows + bool(remainder)
    top, left = rng.randrange(height - box_h + 1), rng.randrange(width - box_w + 1)
    array = np.zeros((height, width), dtype=np.bool_)
    array[top : top + full_rows, left : left + box_w] = True
    if remainder:
        array[top + full_rows, left : left + remainder] = True
    return torch.from_numpy(array).unsqueeze(0)
