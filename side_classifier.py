"""
Front/back classification (Section 14).

Uses layout evidence on the CANONICAL card, not filename/order:
  - front: large photo block on one side (high local variance / skin-tone-ish
    region) + a long digit strip near the bottom
  - back: PDF417-style barcode block (very high horizontal edge density in a
    small region) and no large photo block

This is a coarse heuristic classifier meant to be replaced/augmented by a
trained classifier once labeled samples exist; it exposes its evidence so
the decision is inspectable (Section 40), and returns UNKNOWN with low
confidence rather than guessing when evidence is weak (Section 16 spirit).
"""
import cv2
import numpy as np
from . import config


def _photo_block_score(canonical_bgr: np.ndarray) -> float:
    h, w = canonical_bgr.shape[:2]
    # photo sits in left ~25% for this layout prior; check color variance there
    region = canonical_bgr[int(0.25 * h):int(0.95 * h), 0:int(0.26 * w)]
    if region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat_std = float(np.std(hsv[:, :, 1]))
    val_std = float(np.std(hsv[:, :, 2]))
    return min((sat_std + val_std) / 120.0, 1.0)


def _barcode_block_score(canonical_bgr: np.ndarray) -> float:
    h, w = canonical_bgr.shape[:2]
    region = canonical_bgr[int(0.55 * h):int(0.98 * h), int(0.5 * w):int(0.98 * w)]
    if region.size == 0:
        return 0.0
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edge_density = float(np.mean(np.abs(sobelx) > 40))
    return min(edge_density * 3.0, 1.0)


def classify_side(canonical_bgr: np.ndarray) -> dict:
    photo_score = _photo_block_score(canonical_bgr)
    barcode_score = _barcode_block_score(canonical_bgr)

    front_signal = photo_score - 0.3 * barcode_score
    back_signal = barcode_score - 0.3 * photo_score

    if front_signal < 0.12 and back_signal < 0.12:
        return {
            "side": config.SIDE_UNKNOWN, "confidence": 0.0,
            "evidence": {"photo_score": photo_score, "barcode_score": barcode_score},
        }

    if front_signal >= back_signal:
        conf = min(0.5 + front_signal, 0.98)
        return {"side": config.SIDE_FRONT, "confidence": conf,
                "evidence": {"photo_score": photo_score, "barcode_score": barcode_score}}
    else:
        conf = min(0.5 + back_signal, 0.98)
        return {"side": config.SIDE_BACK, "confidence": conf,
                "evidence": {"photo_score": photo_score, "barcode_score": barcode_score}}
