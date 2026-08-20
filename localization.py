"""
Dynamic field localization + crop extraction (Sections 16, 19).

Starts from the template's normalized region (converted to canonical-card
pixels -- this is scale-invariant by construction, Section 45), then
optionally refines using a cheap text-anchor pass (looking for the densest
band of dark-on-light strokes near the prior region) so small print
variance doesn't clip characters. Refinement never moves the crop outside
a bounded search window around the prior -- it adjusts, it doesn't relocate
arbitrarily.
"""
import cv2
import numpy as np
from . import config
from .coordinate_systems import BBox
from .templates import Template
from .image_utils import assess_crop_quality


def _refine_vertical_bounds(gray_region: np.ndarray) -> tuple:
    """Find the tightest vertical band containing ink, to reduce clipping /
    excess background (Section 19). Falls back to full region if nothing
    found."""
    if gray_region.size == 0:
        return 0, gray_region.shape[0] if gray_region.ndim else 0
    thresh = cv2.adaptiveThreshold(
        gray_region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )
    row_ink = thresh.sum(axis=1)
    nonzero_rows = np.where(row_ink > row_ink.max() * 0.05)[0] if row_ink.max() > 0 else []
    if len(nonzero_rows) == 0:
        return 0, gray_region.shape[0]
    top = max(int(nonzero_rows[0]) - 4, 0)
    bottom = min(int(nonzero_rows[-1]) + 4, gray_region.shape[0])
    return top, bottom


def localize_fields(canonical_bgr: np.ndarray, template: Template) -> dict:
    """
    Returns {field_name: {"bbox": BBox, "crop": np.ndarray, "crop_quality": {...},
                           "localization_confidence": float}}
    """
    h, w = canonical_bgr.shape[:2]
    results = {}

    for spec in template.fields:
        prior = spec.region.to_canonical(card_w=w, card_h=h)
        x0 = max(int(prior.x), 0)
        y0 = max(int(prior.y), 0)
        x1 = min(int(prior.x + prior.w), w)
        y1 = min(int(prior.y + prior.h), h)
        if x1 <= x0 or y1 <= y0:
            results[spec.name] = {
                "bbox": None, "crop": None,
                "crop_quality": {"acceptable": False, "issues": ["degenerate_region"]},
                "localization_confidence": 0.0,
            }
            continue

        region = canonical_bgr[y0:y1, x0:x1]
        localization_confidence = 0.8  # prior-only baseline; template not yet calibrated

        if spec.field_type in ("arabic_text", "numeric", "arabic_numeric") and region.size > 0:
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            top, bottom = _refine_vertical_bounds(gray_region)
            if bottom - top >= config.MIN_FIELD_CROP_HEIGHT_PX:
                # pad a little so we don't clip ascenders/descenders
                pad = max(3, int(0.08 * (bottom - top)))
                top = max(top - pad, 0)
                bottom = min(bottom + pad, region.shape[0])
                region = region[top:bottom]
                y0_final, y1_final = y0 + top, y0 + bottom
                localization_confidence = 0.9
            else:
                y0_final, y1_final = y0, y1
        else:
            y0_final, y1_final = y0, y1

        final_bbox = BBox(
            x=x0, y=y0_final, w=(x1 - x0), h=(y1_final - y0_final),
            coordinate_space=config.SPACE_CANONICAL,
        )
        crop_quality = assess_crop_quality(region)
        results[spec.name] = {
            "bbox": final_bbox,
            "crop": region,
            "crop_quality": crop_quality,
            "localization_confidence": localization_confidence,
            "field_type": spec.field_type,
            "required": spec.required,
        }

    return results
