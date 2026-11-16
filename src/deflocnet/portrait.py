"""Optional single-photo crop that approximates the HDTF face framing."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image, ImageOps

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_BYTES = 232_589

# On 12 local HDTF crop previews, the detector's median face-box height was
# 0.590 of the crop, with median center (0.495, 0.502). The dataset uses
# axis-aligned per-clip crop windows. One photo cannot recover those windows.
_FACE_FRACTION = 0.60
_NO_FACE = "No face detected. Upload a photo with one clearly visible face."


@dataclass(frozen=True)
class PortraitResult:
    image: Image.Image
    crop_box: tuple[int, int, int, int]
    face_box: tuple[float, float, float, float]
    score: float


def _square_box(face_box, width, height):
    left, top, right, bottom = face_box
    side = min(width, height, math.ceil(max(right - left, bottom - top) / _FACE_FRACTION))
    x = min(max(round((left + right - side) / 2), 0), width - side)
    y = min(max(round((top + bottom - side) / 2), 0), height - side)
    return x, y, x + side, y + side


class PortraitCrop:
    """CPU YuNet detector using an operator-provided OpenCV Zoo model file.

    ``opencv-python-headless`` with ``FaceDetectorYN`` is optional until this
    class is constructed. Calling an instance returns 256 × 256 PIL RGB.
    Use ``crop`` to also get the (left, top, right, bottom) boxes in the
    EXIF-oriented source image.
    """

    def __init__(self, detector_path: str | Path):
        path = Path(detector_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Provide the local OpenCV Zoo {YUNET_FILENAME} model file.")
        if path.stat().st_size != YUNET_BYTES or hashlib.sha256(path.read_bytes()).hexdigest() != YUNET_SHA256:
            raise ValueError(f"The detector file must be the official {YUNET_FILENAME} binary.")
        try:
            import cv2
        except (ImportError, OSError) as exc:
            raise RuntimeError("Install opencv-python-headless to enable automatic portrait cropping.") from exc
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("Install a recent opencv-python-headless with FaceDetectorYN support.")
        try:
            self._detector = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.85, 0.3)
        except cv2.error as exc:
            raise RuntimeError("Could not initialize YuNet. Upgrade opencv-python-headless.") from exc
        self._cv2 = cv2
        self._lock = Lock()

    def _detect(self, image):
        for limit in (512, 320):
            if limit == 320 and max(image.size) <= limit:
                break
            scale = min(1.0, limit / max(image.size))
            size = tuple(max(1, round(axis * scale)) for axis in image.size)
            sample = image.resize(size, Image.Resampling.BICUBIC) if size != image.size else image
            bgr = np.ascontiguousarray(np.asarray(sample)[..., ::-1])
            try:
                with self._lock:
                    self._detector.setInputSize(size)
                    _, detections = self._detector.detect(bgr)
            except self._cv2.error as exc:
                raise RuntimeError("YuNet could not process the photo. Check the OpenCV installation.") from exc
            if detections is not None and len(detections):
                if len(detections) > 1:
                    raise ValueError("More than one face detected. Upload a photo containing one face.")
                return detections[0], size
        raise ValueError(_NO_FACE)

    def crop(self, image: Image.Image) -> PortraitResult:
        if min(image.size) < 32 or image.width * image.height > 24_000_000:
            raise ValueError("Use a photo at least 32 pixels on each side and under 24 megapixels.")
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        detection, size = self._detect(rgb)
        x, y, width, height = map(float, detection[:4])
        sx, sy = rgb.width / size[0], rgb.height / size[1]
        face_box = (
            max(0.0, x * sx),
            max(0.0, y * sy),
            min(float(rgb.width), (x + width) * sx),
            min(float(rgb.height), (y + height) * sy),
        )
        if not np.isfinite(detection).all() or face_box[0] >= face_box[2] or face_box[1] >= face_box[3]:
            raise ValueError(_NO_FACE)
        crop_box = _square_box(face_box, *rgb.size)
        result = rgb.crop(crop_box).resize((256, 256), Image.Resampling.BICUBIC)
        return PortraitResult(result, crop_box, face_box, float(detection[14]))

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.crop(image).image
