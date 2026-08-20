"""
Field-specific preprocessing (Section 18).

Each function returns a dict of {variant_name: processed_image} so callers
can run OCR against several variants and let candidate ranking decide
(Section 21). Arabic text is deliberately NOT hit with aggressive
thresholding/sharpening by default -- that destroys dots and connecting
strokes (Section 18 warning).
"""
import cv2
import numpy as np
from . import config


def _upscale_if_small(img: np.ndarray, min_h: int = config.MIN_FIELD_CROP_HEIGHT_PX * 2) -> np.ndarray:
    h, w = img.shape[:2]
    if h < min_h:
        scale = min_h / max(h, 1)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return img


def preprocess_nid(crop_bgr: np.ndarray) -> dict:
    """Numeric strip: high contrast, minimal Arabic-specific concerns."""
    img = _upscale_if_small(crop_bgr, 40)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    variant_a = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)  # contrast normalize

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variant_b = clahe.apply(gray)
    variant_b = cv2.adaptiveThreshold(
        variant_b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    variant_c = cv2.bilateralFilter(gray, 5, 50, 50)
    variant_c = cv2.resize(variant_c, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    return {"contrast_norm": variant_a, "clahe_threshold": variant_b, "denoise_upscale": variant_c}


def preprocess_name(crop_bgr: np.ndarray) -> dict:
    """Arabic name: gentle handling to preserve connected strokes/dots."""
    img = _upscale_if_small(crop_bgr, 50)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    variant_a = cv2.fastNlMeansDenoising(gray, h=7)
    variant_a = cv2.normalize(variant_a, None, 0, 255, cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    variant_b = clahe.apply(gray)

    variant_c = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # minimal processing, keep color

    return {"denoise_contrast": variant_a, "clahe_only": variant_b, "rgb_minimal": variant_c}


def preprocess_address(crop_bgr: np.ndarray) -> dict:
    """Multi-line Arabic text: similar gentle approach, larger crop expected."""
    return preprocess_name(crop_bgr)


def preprocess_dob(crop_bgr: np.ndarray) -> dict:
    return preprocess_nid(crop_bgr)


def preprocess_gender(crop_bgr: np.ndarray) -> dict:
    img = _upscale_if_small(crop_bgr, 40)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variant_a = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    variant_b = clahe.apply(gray)
    return {"contrast_norm": variant_a, "clahe_only": variant_b}


def preprocess_barcode(crop_bgr: np.ndarray) -> dict:
    """Barcode decoding wants sharp edges, not denoised softness."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    variant_a = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    variant_b = cv2.filter2D(variant_a, -1, kernel)
    return {"contrast_norm": variant_a, "sharpened": variant_b}


PREPROCESSORS = {
    "national_id": preprocess_nid,
    "name": preprocess_name,
    "address": preprocess_address,
    "date_of_birth": preprocess_dob,
    "gender": preprocess_gender,
    "barcode": preprocess_barcode,
}


def preprocess_field(field_name: str, crop_bgr: np.ndarray, field_type: str = "arabic_text") -> dict:
    fn = PREPROCESSORS.get(field_name)
    if fn:
        return fn(crop_bgr)
    # fallback by type
    if field_type == "numeric":
        return preprocess_nid(crop_bgr)
    if field_type == "barcode":
        return preprocess_barcode(crop_bgr)
    return preprocess_name(crop_bgr)
