"""
Global configuration and constants.
Nothing here should be treated as ground truth about the *content* of a
specific card (no hard-coded per-image pixel coordinates). It defines the
canonical processing space and shared thresholds only.
"""

# --- Canonical card space -----------------------------------------------
# Egyptian ID physical card is ID-1 format: 85.60mm x 53.98mm (ratio ~1.5857)
CANONICAL_CARD_WIDTH = 1600
CANONICAL_CARD_HEIGHT = int(round(CANONICAL_CARD_WIDTH / (85.60 / 53.98)))  # ~1009

# --- Processing canvas (pre-detection normalization) ---------------------
PROCESSING_MAX_DIM = 2000  # longest side after aspect-preserving resize

# --- Quality thresholds ---------------------------------------------------
MIN_CARD_AREA_RATIO = 0.05       # card must occupy at least 5% of processing canvas
BLUR_LAPLACIAN_MIN = 60.0        # below this -> flagged as blurry
MIN_FIELD_CROP_HEIGHT_PX = 18    # crops shorter than this get upscaled

# --- Confidence thresholds ------------------------------------------------
LOW_CONFIDENCE_OCR = 0.55
LOW_CONFIDENCE_LOCALIZATION = 0.5

# --- Coordinate space identifiers -----------------------------------------
SPACE_ORIGINAL = "original_image"
SPACE_PROCESSING = "processing_canvas"
SPACE_CANONICAL = "canonical_card"
SPACE_FIELD_CROP = "field_crop"

# --- Sides -----------------------------------------------------------------
SIDE_FRONT = "front"
SIDE_BACK = "back"
SIDE_UNKNOWN = "unknown"

# --- Status vocabulary (Section 52 distinction) ---------------------------
STATUS_DETECTED = "DETECTED"           # text found, not yet parsed
STATUS_EXTRACTED = "EXTRACTED"         # parsed into a field value, not verified
STATUS_LOW_CONFIDENCE = "LOW_CONFIDENCE"
STATUS_VALIDATED = "VALIDATED"         # passed internal structural validation
STATUS_CROSS_VALIDATED = "CROSS_VALIDATED"  # agrees with an independent source
STATUS_VERIFIED = "VERIFIED"           # validated + cross-checked, strongest evidence
STATUS_MISMATCH = "MISMATCH"
STATUS_FAILED = "FAILED"

GOVERNORATE_CODES = {
    "01": "القاهرة",
    "02": "الإسكندرية",
    "03": "بورسعيد",
    "04": "السويس",
    "11": "دمياط",
    "12": "الدقهلية",
    "13": "الشرقية",
    "14": "القليوبية",
    "15": "كفر الشيخ",
    "16": "الغربية",
    "17": "المنوفية",
    "18": "البحيرة",
    "19": "الإسماعيلية",
    "21": "الجيزة",
    "22": "بني سويف",
    "23": "الفيوم",
    "24": "المنيا",
    "25": "أسيوط",
    "26": "سوهاج",
    "27": "قنا",
    "28": "أسوان",
    "29": "الأقصر",
    "31": "البحر الأحمر",
    "32": "الوادي الجديد",
    "33": "مطروح",
    "34": "شمال سيناء",
    "35": "جنوب سيناء",
    "88": "خارج مصر (مواليد خارج جمهورية مصر العربية)",
}
