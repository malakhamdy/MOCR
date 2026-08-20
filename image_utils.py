"""
Image normalization + quality assessment.

Key rule (Section 7 / 13): never stretch width/height independently.
Aspect ratio is always preserved via resize + letterbox padding, and the
transform used is recorded so coordinates can be mapped back to the
original image for visualization only (never as the localization source
of truth -- see Section 8/11).
"""
from dataclasses import dataclass
import cv2
import numpy as np

from . import config


@dataclass
class NormalizationTransform:
    original_width: int
    original_height: int
    processing_width: int
    processing_height: int
    scale: float
    pad_x: int
    pad_y: int

    def to_original(self, x: float, y: float):
        """Map a point in processing-canvas space back to original-image space."""
        ox = (x - self.pad_x) / self.scale
        oy = (y - self.pad_y) / self.scale
        return ox, oy

    def to_processing(self, x: float, y: float):
        return x * self.scale + self.pad_x, y * self.scale + self.pad_y

    def as_dict(self):
        return {
            "original_width": self.original_width,
            "original_height": self.original_height,
            "processing_width": self.processing_width,
            "processing_height": self.processing_height,
            "scale": self.scale,
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
        }


def normalize_image(image_bgr: np.ndarray, max_dim: int = config.PROCESSING_MAX_DIM):
    """
    Aspect-ratio-preserving resize into a square-ish letterboxed canvas.
    Returns (processing_image, transform, original_image_copy).
    The original image is returned untouched (never overwritten -- Section 7).
    """
    original = image_bgr.copy()
    h, w = original.shape[:2]

    scale = min(max_dim / w, max_dim / h, 1.0) if max(w, h) > max_dim else max_dim / max(w, h)
    # keep scale reasonable; never upscale huge factors accidentally
    scale = min(scale, 4.0)

    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(original, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    canvas = np.full((max_dim, max_dim, 3), 255, dtype=np.uint8)
    pad_x = (max_dim - new_w) // 2
    pad_y = (max_dim - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    transform = NormalizationTransform(
        original_width=w, original_height=h,
        processing_width=max_dim, processing_height=max_dim,
        scale=scale, pad_x=pad_x, pad_y=pad_y,
    )
    return canvas, transform, original


def laplacian_blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_image_quality(image_bgr: np.ndarray) -> dict:
    """Coarse whole-image quality gate (Section 37)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = laplacian_blur_score(gray)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # crude glare estimate: fraction of near-white pixels
    glare_ratio = float(np.mean(gray > 245))

    issues = []
    if blur < config.BLUR_LAPLACIAN_MIN:
        issues.append("image_is_blurry")
    if brightness < 40:
        issues.append("image_too_dark")
    if brightness > 230:
        issues.append("image_too_bright_or_overexposed")
    if glare_ratio > 0.08:
        issues.append("possible_glare")
    if contrast < 15:
        issues.append("low_contrast")

    return {
        "blur_score": blur,
        "brightness": brightness,
        "contrast": contrast,
        "glare_ratio": glare_ratio,
        "issues": issues,
        "acceptable": len(issues) == 0,
    }


def assess_crop_quality(crop_bgr: np.ndarray) -> dict:
    if crop_bgr is None or crop_bgr.size == 0:
        return {"acceptable": False, "issues": ["empty_crop"]}
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    h, w = gray.shape[:2]
    blur = laplacian_blur_score(gray)
    issues = []
    if h < config.MIN_FIELD_CROP_HEIGHT_PX:
        issues.append("crop_too_small")
    if blur < config.BLUR_LAPLACIAN_MIN * 0.5:
        issues.append("crop_blurry")
    return {
        "width": w, "height": h, "blur_score": blur,
        "issues": issues, "acceptable": len(issues) == 0,
    }
