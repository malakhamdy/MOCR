"""
Barcode / PDF417 handling (Section 31). Separate from OCR entirely.
"""
import cv2
import numpy as np

_PYZBAR_AVAILABLE = True
try:
    from pyzbar.pyzbar import decode as _zbar_decode
except Exception:
    _PYZBAR_AVAILABLE = False

BARCODE_NOT_DETECTED = "barcode_not_detected"
BARCODE_DETECTED = "barcode_detected"
BARCODE_NOT_DECODED = "barcode_not_decoded"
BARCODE_DECODED = "barcode_decoded"


def detect_barcode_region_present(crop_bgr: np.ndarray) -> bool:
    """Cheap heuristic: PDF417/barcode regions have very high local
    horizontal-gradient density compared to plain text."""
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edge_density = float(np.mean(np.abs(sobelx) > 40))
    return edge_density > 0.15


def decode_barcode(crop_bgr: np.ndarray) -> dict:
    """
    Returns a dict with status + decoded payload if available.
    Never invents decoded content if pyzbar isn't installed or fails
    (Section 31: do not assume/ invent decoded content).
    """
    present = detect_barcode_region_present(crop_bgr)
    if not present:
        return {"status": BARCODE_NOT_DETECTED, "payload": None, "symbology": None}

    if not _PYZBAR_AVAILABLE:
        return {
            "status": BARCODE_DETECTED, "payload": None, "symbology": None,
            "note": "pyzbar not installed in this environment; region detected but not decoded.",
        }

    try:
        results = _zbar_decode(crop_bgr)
    except Exception as e:
        return {"status": BARCODE_DETECTED, "payload": None, "symbology": None,
                "note": f"decode attempt raised: {e}"}

    if not results:
        return {"status": BARCODE_NOT_DECODED, "payload": None, "symbology": None}

    r = results[0]
    try:
        payload = r.data.decode("utf-8", errors="replace")
    except Exception:
        payload = repr(r.data)
    return {"status": BARCODE_DECODED, "payload": payload, "symbology": r.type}
