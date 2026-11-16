"""Load operator-selected DeFLOCNet checkpoints and compare completed images."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
import pickle
from threading import Lock

import numpy as np
from PIL import Image
import torch

from .networks import build_generator, predict_generator
from .runtime import Config, source_hashes


IMAGE_SIZE = (256, 256)
_LOAD_LOCK = Lock()


def _load_checkpoint(path, expected_sha256):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected a local checkpoint file: {path}")
    # Training saves NumPy's MT19937 state, whose keys are a uint32 array.
    # Accept only the globals needed for that state and never fall back to pickle.
    reconstruct = np.core.multiarray._reconstruct if np.__version__.startswith("1.") else np._core.multiarray._reconstruct
    rng_globals = [np.ndarray, np.dtype, type(np.dtype("uint32")),
                   (reconstruct, "numpy.core.multiarray._reconstruct"),
                   (reconstruct, "numpy._core.multiarray._reconstruct")]
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
        checksum = digest.hexdigest()
        if expected_sha256 is not None and expected_sha256 != checksum:
            raise ValueError(f"Checkpoint SHA256 mismatch: {path.name}")
        stream.seek(0)
        try:
            with _LOAD_LOCK, torch.serialization.safe_globals(rng_globals):
                checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as exc:
            raise ValueError(f"Checkpoint could not be loaded with the restricted loader: {path.name}") from exc
    return checkpoint, checksum, path.name


def _load_generator(path, expected_sha256, device, current_source):
    checkpoint, checksum, filename = _load_checkpoint(path, expected_sha256)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("format") != "deflocnet_modern_v1":
        raise ValueError(f"Expected a deflocnet_modern_v1 checkpoint: {filename}")
    step = checkpoint.get("step")
    if type(step) is not int or step < 0:
        raise ValueError(f"Invalid training step: {filename}")
    config = Config(**checkpoint["config"])
    source = checkpoint.get("source")
    modules = ["networks.py"]
    if config.architecture in {"sd_controls_v1", "sd_color_v2"}:
        modules.append("sd_networks.py")
    if config.architecture == "sd_color_v2":
        modules.append("sd_color_networks.py")
    # Training/evaluation scripts may differ between runs. The model's source
    # must match so equal state keys cannot silently load changed semantics.
    if not isinstance(source, Mapping) or any(source.get(name) != current_source[name] for name in modules):
        raise ValueError(f"Generator source differs from the checkpoint: {filename}")
    state = checkpoint.get("generator")
    if not isinstance(state, Mapping) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"Invalid generator state: {filename}")
    del checkpoint  # Discard training-only tensors before constructing the generator.
    model = build_generator(config).float().eval().requires_grad_(False)
    model.load_state_dict(state, strict=True)
    return model.to(device), {"step": step, "architecture": config.architecture,
                              "sha256": checksum, "filename": filename}


def _convert(image, mode, name):
    if not isinstance(image, Image.Image) or image.size != IMAGE_SIZE:
        raise ValueError(f"{name} must be a 256×256 image")
    return image.convert(mode)


def _rgb_tensor(image, device):
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).unsqueeze(0).to(
        device=device, dtype=torch.float32) / 255


def _color_tensor(color, device):
    if not isinstance(color, np.ndarray):
        return _rgb_tensor(_convert(color, "RGB", "Color"), device)
    if (color.shape != (*IMAGE_SIZE, 3) or color.dtype != np.float32 or
            not np.isfinite(color).all() or color.min() < 0 or color.max() > 1):
        raise ValueError("RTV color must be 256×256 RGB float32 in [0, 1]")
    return torch.from_numpy(np.array(color, copy=True)).permute(2, 0, 1).unsqueeze(0).to(device)


def _binary_mask(mask, name):
    return _convert(mask, "L", name).point([0] * 128 + [255] * 128)


def _mask_tensor(mask, device):
    return torch.from_numpy(np.asarray(mask) != 0).to(device)[None, None]


class DemoInference:
    """Serve one or two fixed models on a common image, mask and noise sample."""

    def __init__(self, checkpoints: Mapping[str, str | Path], device: str = "cpu", *,
                 expected_sha256: Mapping[str, str] | None = None):
        if not 1 <= len(checkpoints) <= 2:
            raise ValueError("Configure one or two checkpoints")
        expected_sha256 = dict(expected_sha256 or {})
        if expected_sha256.keys() - checkpoints.keys():
            raise ValueError("Expected SHA256 keys must match configured checkpoints")
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Use a CPU or CUDA device")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        self._lock = Lock()
        self._models, self._metadata = {}, {}
        current_source = source_hashes()
        for label, path in checkpoints.items():
            model, metadata = _load_generator(path, expected_sha256.get(label), self.device, current_source)
            self._models[label], self._metadata[label] = model, metadata

    def metadata(self) -> dict[str, dict]:
        return {label: dict(metadata) for label, metadata in self._metadata.items()}

    @torch.inference_mode()
    def predict(self, image: Image.Image, mask: Image.Image, *, sketch: Image.Image | None = None,
                color: Image.Image | np.ndarray | None = None, color_mask: Image.Image | None = None,
                seed: int = 2019, model_names: Sequence[str] | None = None) -> dict[str, Image.Image]:
        """White mask pixels are selected. Sketch is white strokes on black."""
        names = tuple(self._models) if model_names is None else tuple(model_names)
        if not names or len(names) != len(set(names)) or any(name not in self._models for name in names):
            raise ValueError("Select one or both configured models.")
        if (color is None) != (color_mask is None):
            raise ValueError("Supply color and its explicit color mask together")
        if type(seed) is not int or not -(2**63) <= seed < 2**64:
            raise ValueError("Seed is outside the supported integer range")
        rgb = _convert(image, "RGB", "Image")
        hole_image = _binary_mask(mask, "Hole mask")
        sketch_image = _convert(sketch, "RGB", "Sketch") if sketch is not None else None
        selected_image = _binary_mask(color_mask, "Color mask") if color_mask is not None else None
        if hole_image.getbbox() is None:
            return {label: rgb.copy() for label in names}

        # Each request keeps one model's activations live at a time. Separate
        # local RNGs avoid interference from other requests or torch users.
        with self._lock, torch.autocast(device_type=self.device.type, enabled=False):
            image_tensor = _rgb_tensor(rgb, self.device) * 2 - 1
            hole = _mask_tensor(hole_image, self.device)
            empty_rgb = torch.zeros_like(image_tensor)
            sketch_tensor = (_rgb_tensor(sketch_image, self.device) * hole
                             if sketch_image is not None else empty_rgb)
            selected = (_mask_tensor(selected_image, self.device) & hole
                        if selected_image is not None else torch.zeros_like(hole))
            color_tensor = ((_color_tensor(color, self.device) * 2 - 1) * selected
                            if color is not None else empty_rgb)
            rng = torch.Generator(device=self.device).manual_seed(seed)
            noise = torch.rand(image_tensor.shape, dtype=torch.float32, device=self.device, generator=rng)
            results = {}
            for label in names:
                model = self._models[label]
                raw = predict_generator(model, image_tensor, hole, sketch_tensor, color_tensor, noise, selected)
                if raw.shape != image_tensor.shape or not bool(torch.isfinite(raw).all()):
                    raise ValueError(f"Model {label} returned invalid RGB output")
                pixels = ((raw[0].float() + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                prediction = Image.fromarray(np.rint(pixels * 255).astype(np.uint8))
                # Preserve input bytes directly rather than round-tripping known pixels.
                results[label] = Image.composite(prediction, rgb, hole_image)
        return results
