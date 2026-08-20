"""
Lightweight document-forgery / tamper heuristics.

*** NEW MODULE — added on top of the existing pipeline, nothing in this
    file is used by or changes core/pipeline.py, card_detection.py,
    localization.py, ocr_engine.py, nid_validator.py, or
    cross_field_validator.py. Those remain exactly as they were. ***

Honest scope statement (matching this project's existing "no fake success
claims" convention in the README):

  - There is no trained forgery-classification model here. Real document
    forensics (screen-recapture detection, print/scan detection, ICC/PRNU
    analysis, trained tamper-localization networks) needs labeled data and
    dedicated models this project does not have.
  - What IS implemented below is a small set of classical, unsupervised
    image-forensics heuristics that are cheap to run and have *some* signal,
    but produce false positives/negatives regularly:
      1. Error Level Analysis (ELA)     -> localized JPEG re-compression
      2. Copy-move detection (ORB)      -> duplicated regions pasted elsewhere
      3. Screen/print recapture check   -> periodic moire-like FFT energy
      4. Stroke-width consistency       -> mismatched font/ink across fields
      5. Photo-block edge halo check    -> a pasted-in photo's cut edge
  - Each check returns its raw signal + a boolean flag + a short reason
    string. Nothing here should be treated as a legal/forensic verdict.
    Treat the combined "status" as a *screening* signal, not a determination
    that a document is fake.
"""
from dataclasses import dataclass, asdict
import cv2
import numpy as np

STATUS_LIKELY_AUTHENTIC = "LIKELY_AUTHENTIC"
STATUS_NO_STRONG_SIGNAL = "NO_STRONG_SIGNAL"
STATUS_SUSPICIOUS = "SUSPICIOUS"
STATUS_LIKELY_MANIPULATED = "LIKELY_MANIPULATED"
STATUS_INCONCLUSIVE = "INCONCLUSIVE"  # not enough material to run checks


@dataclass
class CheckResult:
    name: str
    ran: bool
    score: float          # 0.0 (no signal) .. 1.0 (strong signal)
    flagged: bool
    detail: str
    reason: str = ""

    def as_dict(self):
        return asdict(self)


def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------
# 1. Error Level Analysis
# --------------------------------------------------------------------------
def _error_level_analysis(original_bgr: np.ndarray, quality: int = 90) -> CheckResult:
    try:
        ok, encoded = cv2.imencode(".jpg", original_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return CheckResult("error_level_analysis", False, 0.0, False, "encode_failed")
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if recompressed is None or recompressed.shape != original_bgr.shape:
            return CheckResult("error_level_analysis", False, 0.0, False, "shape_mismatch")

        diff = cv2.absdiff(original_bgr, recompressed).astype(np.float32)
        ela = diff.mean(axis=2)  # single-channel ELA response

        h, w = ela.shape
        block = 24
        block_means = []
        for y in range(0, h - block, block):
            for x in range(0, w - block, block):
                block_means.append(float(ela[y:y + block, x:x + block].mean()))
        if len(block_means) < 8:
            return CheckResult("error_level_analysis", False, 0.0, False, "too_small_for_blocks")

        block_means = np.array(block_means)
        median = float(np.median(block_means))
        p95 = float(np.percentile(block_means, 95))
        # A large gap between the typical block and the hottest blocks
        # suggests a localized region was edited/re-saved separately from
        # the rest of the image (whole-image recompression is uniform).
        ratio = p95 / max(median, 1e-6)
        score = float(np.clip((ratio - 3.0) / 7.0, 0.0, 1.0))
        flagged = ratio > 5.0
        detail = f"ela_p95_to_median_ratio={ratio:.2f}"
        reason = "Localized recompression signature inconsistent with the rest of the image." if flagged else ""
        return CheckResult("error_level_analysis", True, score, flagged, detail, reason)
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult("error_level_analysis", False, 0.0, False, f"error:{exc}")


# --------------------------------------------------------------------------
# 2. Copy-move detection
# --------------------------------------------------------------------------
def _copy_move_detection(canonical_bgr: np.ndarray) -> CheckResult:
    try:
        gray = _to_gray(canonical_bgr)
        orb = cv2.ORB_create(nfeatures=1500)
        kp, desc = orb.detectAndCompute(gray, None)
        if desc is None or len(kp) < 20:
            return CheckResult("copy_move_detection", False, 0.0, False, "too_few_keypoints")

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(desc, desc, k=3)  # k=3 because best match is always self

        h, w = gray.shape[:2]
        min_spatial_dist = 0.12 * max(h, w)  # ignore near-identical/adjacent texture
        suspicious_pairs = 0
        for m in matches:
            for cand in m[1:]:  # skip index 0 (self-match, distance 0)
                if cand.distance > 28:  # tight Hamming distance = near-identical patch
                    continue
                p1 = np.array(kp[cand.queryIdx].pt)
                p2 = np.array(kp[cand.trainIdx].pt)
                if np.linalg.norm(p1 - p2) > min_spatial_dist:
                    suspicious_pairs += 1

        score = float(np.clip(suspicious_pairs / 40.0, 0.0, 1.0))
        flagged = suspicious_pairs >= 12
        detail = f"suspicious_duplicate_keypoint_pairs={suspicious_pairs}"
        reason = "Found repeated, spatially-distant patches that may indicate a pasted/cloned region." if flagged else ""
        return CheckResult("copy_move_detection", True, score, flagged, detail, reason)
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult("copy_move_detection", False, 0.0, False, f"error:{exc}")


# --------------------------------------------------------------------------
# 3. Screen / print recapture check (moire-like periodic energy)
# --------------------------------------------------------------------------
def _screen_recapture_check(original_bgr: np.ndarray) -> CheckResult:
    try:
        gray = _to_gray(original_bgr).astype(np.float32)
        gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
        f = np.fft.fftshift(np.fft.fft2(gray))
        mag = np.log(np.abs(f) + 1.0)

        cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
        # Exclude the low-frequency DC blob; look at the mid-frequency ring
        # where screen refresh / pixel-grid moire tends to show up as
        # unusually sharp, spatially concentrated peaks.
        yy, xx = np.ogrid[:mag.shape[0], :mag.shape[1]]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        ring_mask = (r > 40) & (r < 180)
        ring_vals = mag[ring_mask]
        if ring_vals.size < 100:
            return CheckResult("screen_recapture_check", False, 0.0, False, "insufficient_spectrum")

        mean_energy = float(ring_vals.mean())
        peak_energy = float(np.percentile(ring_vals, 99.5))
        peakiness = peak_energy / max(mean_energy, 1e-6)
        score = float(np.clip((peakiness - 1.6) / 1.4, 0.0, 1.0))
        flagged = peakiness > 2.2
        detail = f"mid_frequency_peakiness={peakiness:.2f}"
        reason = "Mid-frequency spectral peaks consistent with a photographed screen or moire pattern." if flagged else ""
        return CheckResult("screen_recapture_check", True, score, flagged, detail, reason)
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult("screen_recapture_check", False, 0.0, False, f"error:{exc}")


# --------------------------------------------------------------------------
# 4. Stroke-width consistency across OCR'd text fields
# --------------------------------------------------------------------------
def _median_stroke_width(gray_crop: np.ndarray):
    if gray_crop is None or gray_crop.size == 0:
        return None
    thresh = cv2.adaptiveThreshold(
        gray_crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )
    if thresh.sum() == 0:
        return None
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
    ink_widths = dist[thresh > 0] * 2.0
    if ink_widths.size == 0:
        return None
    return float(np.median(ink_widths))


def _font_consistency_check(field_crops: dict) -> CheckResult:
    try:
        widths = {}
        for name, crop in (field_crops or {}).items():
            if crop is None or crop.size == 0:
                continue
            gray = _to_gray(crop)
            mw = _median_stroke_width(gray)
            if mw is not None and mw > 0:
                widths[name] = mw

        if len(widths) < 2:
            return CheckResult("font_consistency_check", False, 0.0, False, "not_enough_text_fields")

        values = np.array(list(widths.values()))
        median_w = float(np.median(values))
        max_ratio = float(np.max(np.abs(values - median_w)) / max(median_w, 1e-6))
        score = float(np.clip((max_ratio - 0.35) / 0.65, 0.0, 1.0))
        flagged = max_ratio > 0.6
        outlier = max(widths, key=lambda k: abs(widths[k] - median_w))
        detail = f"stroke_widths={ {k: round(v, 2) for k, v in widths.items()} }, max_deviation_ratio={max_ratio:.2f}"
        reason = (f"Field '{outlier}' has a noticeably different stroke width/ink weight than the other "
                  f"printed fields, which can indicate it was edited or re-printed separately.") if flagged else ""
        return CheckResult("font_consistency_check", True, score, flagged, detail, reason)
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult("font_consistency_check", False, 0.0, False, f"error:{exc}")


# --------------------------------------------------------------------------
# 5. Photo-block edge halo check
# --------------------------------------------------------------------------
def _photo_boundary_check(canonical_bgr: np.ndarray, photo_crop) -> CheckResult:
    try:
        if photo_crop is None or photo_crop.size == 0:
            return CheckResult("photo_boundary_check", False, 0.0, False, "no_photo_crop")
        gray = _to_gray(photo_crop)
        h, w = gray.shape[:2]
        if h < 20 or w < 20:
            return CheckResult("photo_boundary_check", False, 0.0, False, "photo_crop_too_small")

        edges = cv2.Canny(gray, 60, 160)
        border = 3
        border_mask = np.zeros_like(edges, dtype=bool)
        border_mask[:border, :] = True
        border_mask[-border:, :] = True
        border_mask[:, :border] = True
        border_mask[:, -border:] = True

        border_edge_density = float(edges[border_mask].mean() / 255.0)
        interior_edge_density = float(edges[~border_mask].mean() / 255.0)
        ratio = border_edge_density / max(interior_edge_density, 0.02)

        score = float(np.clip((ratio - 2.0) / 4.0, 0.0, 1.0))
        flagged = ratio > 3.5
        detail = f"border_to_interior_edge_ratio={ratio:.2f}"
        reason = "Unusually sharp, continuous edge around the photo region, consistent with a pasted-in photo." if flagged else ""
        return CheckResult("photo_boundary_check", True, score, flagged, detail, reason)
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult("photo_boundary_check", False, 0.0, False, f"error:{exc}")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
_CHECK_WEIGHTS = {
    "error_level_analysis": 0.25,
    "copy_move_detection": 0.25,
    "screen_recapture_check": 0.15,
    "font_consistency_check": 0.20,
    "photo_boundary_check": 0.15,
}


def analyze_document_forgery(original_bgr: np.ndarray, canonical_bgr, field_crops: dict) -> dict:
    """
    Runs all heuristic checks and combines them into a single screening
    result. `canonical_bgr` and `field_crops` may be None/empty if card
    detection or localization failed upstream -- checks that need them are
    simply skipped (not treated as evidence either way).

    Returns a dict: {status, suspicion_score, checks: {...}, reasons: [...],
    disclaimer}.
    """
    checks = [_error_level_analysis(original_bgr)]

    if canonical_bgr is not None and canonical_bgr.size > 0:
        checks.append(_copy_move_detection(canonical_bgr))
    else:
        checks.append(CheckResult("copy_move_detection", False, 0.0, False, "no_canonical_card"))

    checks.append(_screen_recapture_check(original_bgr))
    checks.append(_font_consistency_check(field_crops))

    photo_crop = (field_crops or {}).get("photo")
    checks.append(_photo_boundary_check(canonical_bgr, photo_crop))

    ran_checks = [c for c in checks if c.ran]
    if not ran_checks:
        return {
            "status": STATUS_INCONCLUSIVE,
            "suspicion_score": None,
            "checks": {c.name: c.as_dict() for c in checks},
            "reasons": [],
            "disclaimer": (
                "No forgery heuristics could run on this image (card/fields not "
                "usable). This is not evidence of authenticity or tampering."
            ),
        }

    total_weight = sum(_CHECK_WEIGHTS.get(c.name, 0.1) for c in ran_checks)
    weighted_score = sum(c.score * _CHECK_WEIGHTS.get(c.name, 0.1) for c in ran_checks) / max(total_weight, 1e-6)

    if weighted_score < 0.25:
        status = STATUS_LIKELY_AUTHENTIC
    elif weighted_score < 0.45:
        status = STATUS_NO_STRONG_SIGNAL
    elif weighted_score < 0.7:
        status = STATUS_SUSPICIOUS
    else:
        status = STATUS_LIKELY_MANIPULATED

    reasons = [c.reason for c in checks if c.flagged and c.reason]

    return {
        "status": status,
        "suspicion_score": round(float(weighted_score), 3),
        "checks": {c.name: c.as_dict() for c in checks},
        "reasons": reasons,
        "disclaimer": (
            "Experimental, unsupervised heuristics (ELA, copy-move, moire/recapture, "
            "stroke-width, photo-edge). Not a certified forensic tool and not trained "
            "on labeled real/fake Egyptian ID samples -- treat as a screening signal "
            "to route for human review, never as an automatic accept/reject decision."
        ),
    }
