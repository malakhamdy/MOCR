"""
Card detection + perspective rectification (Sections 9-10).

detect_card_corners(): finds the best quadrilateral candidate for the
physical card in the processing canvas.

rectify_card(): applies a homography to map the detected quad onto the
canonical card rectangle (config.CANONICAL_CARD_WIDTH x HEIGHT).
"""
import cv2
import numpy as np
from . import config


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def detect_card_corners(processing_image_bgr: np.ndarray):
    """
    Returns (corners, confidence, debug_info) where corners is a 4x2
    float32 array (TL,TR,BR,BL) in processing-canvas coordinates, or
    (None, 0.0, debug_info) if no plausible quadrilateral was found.

    Tries, in order:
      1. Standard contour/quad detection (card photographed with visible
         background margin).
      2. A relaxed contour pass with looser polygon-approximation epsilon,
         for cases where the card's outer edge is faint/low-contrast.
      3. A full-frame fallback for already-tightly-cropped card photos,
         where the card fills nearly the whole frame and there is no
         background border for a contour to close against. This is common
         for phone screenshots of a card held up close, or pre-cropped
         datasets -- verified against real sample images where the
         contour-based approach alone found nothing (max contour ~3-4% of
         frame area, well below MIN_CARD_AREA_RATIO).
    """
    h, w = processing_image_bgr.shape[:2]
    canvas_area = w * h

    corners, score, debug = _detect_via_contours(processing_image_bgr, approx_eps=0.02)
    if corners is not None:
        debug["strategy"] = "contour_standard"
        return corners, score, debug

    corners, score, debug2 = _detect_via_contours(processing_image_bgr, approx_eps=0.04)
    debug2["num_contours_standard_pass"] = debug["num_contours"]
    if corners is not None:
        debug2["strategy"] = "contour_relaxed"
        return corners, score, debug2

    corners, score, debug3 = _detect_full_frame_fallback(processing_image_bgr)
    debug3["num_contours_standard_pass"] = debug["num_contours"]
    if corners is not None:
        debug3["strategy"] = "full_frame_fallback"
        return corners, score, debug3

    debug3["strategy"] = "none"
    return None, 0.0, debug3


def _detect_via_contours(processing_image_bgr: np.ndarray, approx_eps: float = 0.02):
    h, w = processing_image_bgr.shape[:2]
    gray = cv2.cvtColor(processing_image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    canvas_area = w * h

    best = None
    best_score = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < canvas_area * config.MIN_CARD_AREA_RATIO:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, approx_eps * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        ordered = _order_corners(approx)
        side_top = np.linalg.norm(ordered[1] - ordered[0])
        side_bottom = np.linalg.norm(ordered[2] - ordered[3])
        side_left = np.linalg.norm(ordered[3] - ordered[0])
        side_right = np.linalg.norm(ordered[2] - ordered[1])
        if min(side_top, side_bottom, side_left, side_right) < 1:
            continue

        est_w = (side_top + side_bottom) / 2
        est_h = (side_left + side_right) / 2
        aspect = est_w / max(est_h, 1e-6)
        target_aspect = config.CANONICAL_CARD_WIDTH / config.CANONICAL_CARD_HEIGHT
        aspect_error = min(
            abs(aspect - target_aspect),
            abs(1 / aspect - target_aspect),
        )
        aspect_score = max(0.0, 1.0 - aspect_error / target_aspect)
        area_score = area / canvas_area

        score = 0.6 * aspect_score + 0.4 * min(area_score / 0.5, 1.0)
        if score > best_score:
            best_score = score
            best = ordered

    debug = {"num_contours": len(contours), "canvas_area": canvas_area, "approx_eps": approx_eps}
    if best is None:
        return None, 0.0, debug
    return best, float(best_score), debug


def _detect_full_frame_fallback(processing_image_bgr: np.ndarray):
    """
    For images where the card already fills (almost) the entire frame and
    no background border exists for contour closure. `processing_image_bgr`
    is the letterboxed square processing canvas (Section 7), so we first
    recover the actual non-padding CONTENT region (the letterbox pad color
    is solid white) before reasoning about aspect ratio -- checking the
    aspect of the full square canvas would always fail, since it's
    square by construction regardless of the original photo's shape.

    Confirms plausibility by checking the content region's aspect ratio is
    reasonably close to an ID card. Returns a lower confidence than a real
    detected quad, since this is an assumption rather than direct evidence.
    """
    h, w = processing_image_bgr.shape[:2]

    # Recover the actual (non-letterbox) content bounding box.
    non_white = np.any(processing_image_bgr < 250, axis=2)
    ys, xs = np.where(non_white)
    if len(xs) == 0 or len(ys) == 0:
        return None, 0.0, {"reason": "no_content_found"}
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    content_w, content_h = (x1 - x0), (y1 - y0)
    if content_w < 10 or content_h < 10:
        return None, 0.0, {"reason": "content_region_too_small"}

    target_aspect = config.CANONICAL_CARD_WIDTH / config.CANONICAL_CARD_HEIGHT
    content_aspect = content_w / content_h
    aspect_error = min(abs(content_aspect - target_aspect), abs((1 / content_aspect) - target_aspect))
    if aspect_error / target_aspect > 0.35:
        return None, 0.0, {"reason": "content_aspect_not_card_like", "content_aspect": content_aspect}

    # small inset so we don't include a thin true background sliver / photo edge halo
    inset_x = int(0.01 * content_w)
    inset_y = int(0.01 * content_h)
    corners = np.array([
        [x0 + inset_x, y0 + inset_y],
        [x1 - inset_x, y0 + inset_y],
        [x1 - inset_x, y1 - inset_y],
        [x0 + inset_x, y1 - inset_y],
    ], dtype="float32")

    confidence = 0.45  # deliberately capped below a real contour-verified detection
    debug = {"reason": "full_frame_assumed", "content_aspect": content_aspect,
              "content_bbox": [x0, y0, x1, y1]}
    return corners, confidence, debug


def rectify_card(processing_image_bgr: np.ndarray, corners: np.ndarray):
    """
    Warp the detected quadrilateral (processing-canvas coords) into the
    canonical card rectangle. Returns (canonical_card_bgr, homography_matrix).
    """
    dst = np.array([
        [0, 0],
        [config.CANONICAL_CARD_WIDTH - 1, 0],
        [config.CANONICAL_CARD_WIDTH - 1, config.CANONICAL_CARD_HEIGHT - 1],
        [0, config.CANONICAL_CARD_HEIGHT - 1],
    ], dtype="float32")

    homography = cv2.getPerspectiveTransform(corners, dst)
    canonical = cv2.warpPerspective(
        processing_image_bgr, homography,
        (config.CANONICAL_CARD_WIDTH, config.CANONICAL_CARD_HEIGHT),
    )
    return canonical, homography


def detect_and_rectify(processing_image_bgr: np.ndarray):
    """
    Convenience wrapper. Returns a dict with corners, confidence, canonical
    card image (or None), and homography (or None).
    """
    corners, confidence, debug = detect_card_corners(processing_image_bgr)
    if corners is None:
        return {
            "corners": None, "confidence": 0.0,
            "canonical_card": None, "homography": None, "debug": debug,
        }
    canonical, homography = rectify_card(processing_image_bgr, corners)
    return {
        "corners": corners.tolist(), "confidence": confidence,
        "canonical_card": canonical, "homography": homography, "debug": debug,
    }
