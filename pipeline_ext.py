"""
API-facing orchestration layer.

*** This module does NOT change core/pipeline.py, core/card_detection.py,
    core/localization.py, core/ocr_engine.py, core/nid_validator.py, or
    core/cross_field_validator.py. It only calls them. ***

run_pipeline() (unmodified) is the single source of truth for the
extraction/validation result. This wrapper additionally re-derives the
canonical (rectified) card + per-field crops -- using the same, unmodified
card_detection / side_classifier / templates / localization functions the
real pipeline uses internally -- purely so the new, separate forgery_check
module has image material to analyze, and so the API can hand the app a
rectified card image to draw the returned bboxes on (all returned bboxes
are already in that canonical-card coordinate space, per core/schema.py).

This does mean card detection/rectification runs twice per request (once
inside run_pipeline, once here). Both calls use the same classical-CV
functions (no ML inference), so the extra cost is small; if that ever
matters, pipeline.py could be extended to optionally return the canonical
card instead -- deliberately left as-is per the "don't touch existing
pipeline logic" constraint.
"""
import cv2
import numpy as np

from . import config
from .pipeline import run_pipeline
from .image_utils import normalize_image
from .card_detection import detect_and_rectify
from .side_classifier import classify_side
from .templates import get_template_for_side
from .localization import localize_fields
from .forgery_check import analyze_document_forgery


def _get_canonical_card_and_crops(image_bgr: np.ndarray):
    """Reuses the existing detection/localization functions unmodified."""
    processing_img, _transform, original_img = normalize_image(image_bgr)
    detection = detect_and_rectify(processing_img)
    canonical_card = detection.get("canonical_card")
    if canonical_card is None:
        return None, {}

    side_info = classify_side(canonical_card)
    if side_info["side"] == config.SIDE_UNKNOWN:
        return canonical_card, {}

    template = get_template_for_side(side_info["side"])
    localized = localize_fields(canonical_card, template)
    crops = {name: info["crop"] for name, info in localized.items() if info.get("crop") is not None}
    return canonical_card, crops


def run_full_analysis(image_bgr: np.ndarray, run_ocr: bool = True, enable_forgery_check: bool = True) -> dict:
    """
    Returns (result_dict, canonical_card_bgr_or_None).

    result_dict is core.schema.PipelineResult.as_dict() (unchanged shape)
    plus one new top-level key: "forgery_analysis".
    """
    pipeline_result = run_pipeline(image_bgr, run_ocr=run_ocr)
    result_dict = pipeline_result.as_dict()

    canonical_card = None
    forgery_result = None

    if enable_forgery_check:
        try:
            canonical_card, field_crops = _get_canonical_card_and_crops(image_bgr)
            forgery_result = analyze_document_forgery(image_bgr, canonical_card, field_crops)
        except Exception as exc:  # pragma: no cover - defensive, must never break the main result
            forgery_result = {
                "status": "INCONCLUSIVE",
                "suspicion_score": None,
                "checks": {},
                "reasons": [],
                "disclaimer": f"Forgery analysis failed to run: {exc}",
            }
    else:
        # Still try to get the canonical card for bbox-overlay purposes even
        # if forgery checks were skipped, but never let this fail the request.
        try:
            canonical_card, _ = _get_canonical_card_and_crops(image_bgr)
        except Exception:
            canonical_card = None

    result_dict["forgery_analysis"] = forgery_result
    return result_dict, canonical_card


def encode_image_base64_jpeg(image_bgr: np.ndarray, quality: int = 85):
    if image_bgr is None:
        return None
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    import base64
    return base64.b64encode(buf.tobytes()).decode("ascii")
