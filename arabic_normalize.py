"""
Arabic normalization (Section 23).

Always returns BOTH raw and normalized -- never overwrite raw with
normalized (Section 1 / 23).
"""
import re
import unicodedata

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
PERSIAN_INDIC_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
WESTERN_DIGITS = "0123456789"

_DIGIT_MAP = {}
for _i, _ch in enumerate(ARABIC_INDIC_DIGITS):
    _DIGIT_MAP[_ch] = WESTERN_DIGITS[_i]
for _i, _ch in enumerate(PERSIAN_INDIC_DIGITS):
    _DIGIT_MAP[_ch] = WESTERN_DIGITS[_i]

TATWEEL = "\u0640"

# common OCR punctuation noise seen around Arabic ID text
_NOISE_CHARS_RE = re.compile(r"[|_\\/~`\^\*]+")


def digits_to_western(text: str) -> str:
    return "".join(_DIGIT_MAP.get(ch, ch) for ch in text)


def strip_tatweel(text: str) -> str:
    return text.replace(TATWEEL, "")


def normalize_arabic_text(raw_text: str) -> str:
    if raw_text is None:
        return ""
    text = unicodedata.normalize("NFKC", raw_text)
    text = strip_tatweel(text)
    text = _NOISE_CHARS_RE.sub(" ", text)
    text = digits_to_western(text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_numeric_field(raw_text: str) -> str:
    """For NID / dates: strip everything that isn't a digit after digit
    conversion (spaces, dashes, OCR noise), preserving raw separately."""
    if raw_text is None:
        return ""
    text = digits_to_western(raw_text)
    text = re.sub(r"[^\d]", "", text)
    return text


def make_field_value(raw_text: str, is_numeric: bool = False) -> dict:
    normalized = normalize_numeric_field(raw_text) if is_numeric else normalize_arabic_text(raw_text)
    return {"raw": raw_text, "normalized": normalized}
