"""Versioned HED guide processing, shared by training and validation.

HED predicts an edge *response*, not a one-pixel drawing. Estimate the ridge
normal from the smoothed response Hessian, suppress non-maxima along that
normal, then thin the thresholded ridge with Zhang-Suen skeletonization.
This is our explicit SciPy/skimage protocol, not a bitwise MATLAB reproduction.
Process the full image before applying a hole or control-enable mask.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, generate_binary_structure, map_coordinates
import torch

LEGACY = "threshold_v1"
THIN = "hed_nms_thin_v1"
MEDIUM = "hed_nms_medium_v1"


def edge_nms(probability: np.ndarray) -> np.ndarray:
    """Bilinear directional NMS at +/-1 pixel, with 1% response tolerance."""
    p = np.asarray(probability, dtype=np.float32)
    if p.ndim != 2 or min(p.shape) < 3:
        raise ValueError("HED probabilities must be a 2D image at least 3x3")
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError("HED probabilities must be finite and in [0,1]")
    smooth = gaussian_filter(p, sigma=1.0, mode="reflect")
    gy, gx = np.gradient(smooth)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    cross = (gxy + gyx) * 0.5
    # The eigenvector of the most negative curvature points across a ridge.
    angle = 0.5 * np.arctan2(2 * cross, gxx - gyy) + np.pi / 2
    curvature = 0.5 * (gxx + gyy - np.hypot(gxx - gyy, 2 * cross))
    dy, dx = np.sin(angle), np.cos(angle)
    yy, xx = np.mgrid[:p.shape[0], :p.shape[1]].astype(np.float32)
    ahead = map_coordinates(p, [yy + dy, xx + dx], order=1, mode="nearest", prefilter=False)
    behind = map_coordinates(p, [yy - dy, xx - dx], order=1, mode="nearest", prefilter=False)
    keep = (curvature < -1e-6) & (p * 1.01 >= ahead) & (p * 1.01 >= behind)
    return np.where(keep, p, 0).astype(np.float32)


@torch.no_grad()
def postprocess_edges(probabilities: torch.Tensor, protocol: str) -> torch.Tensor:
    """Return binary B1HW FP32 guides on the original device, without RNG use."""
    if probabilities.ndim != 4 or probabilities.shape[1] != 1:
        raise ValueError("Expected B1HW HED probabilities")
    if protocol == LEGACY:
        return (probabilities > 0.5).float()
    if protocol not in {THIN, MEDIUM}:
        raise ValueError(f"Unknown HED guide protocol: {protocol}")
    from skimage.morphology import skeletonize

    maps = probabilities.detach().float().cpu().numpy()[:, 0]
    lines = np.stack([skeletonize(edge_nms(p) > 0.5, method="zhang") for p in maps])
    if protocol == MEDIUM:
        # Match the reviewed 2-3px candidate: one symmetric radius-1 cross,
        # applied per image before hole masking, with no batch/channel mixing.
        footprint = generate_binary_structure(2, 1)
        lines = np.stack([binary_dilation(line, structure=footprint, iterations=1) for line in lines])
    return torch.from_numpy(lines[:, None].astype(np.float32)).to(probabilities.device)


@torch.no_grad()
def hed_sketch(hed, rgb: torch.Tensor, protocol: str) -> torch.Tensor:
    return postprocess_edges(hed(rgb), protocol).expand_as(rgb)
