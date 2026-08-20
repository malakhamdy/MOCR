"""
Field parsing (Section 17): turn ranked OCR candidates into semantic field
values. Deliberately conservative -- no regex "fix-ups" that invent digits
or characters (Section 1, Section 36).
"""
import re
from .arabic_normalize import normalize_arabic_text, normalize_numeric_field


def parse_national_id(candidate_text: str) -> str:
    return normalize_numeric_field(candidate_text)


def parse_name(candidate_text: str) -> str:
    return normalize_arabic_text(candidate_text)


def parse_address(candidate_texts: list) -> str:
    """candidate_texts: ordered list of line strings (RTL line order should
    already be handled by the OCR engine's reading order)."""
    lines = [normalize_arabic_text(t) for t in candidate_texts if t and t.strip()]
    return " ".join(lines)


def parse_dob_printed(candidate_text: str) -> str:
    """
    Attempt to parse a printed DOB into ISO format. Accepts common
    separators. Returns None (not a guess) if the pattern doesn't clearly
    resolve to day/month/year -- ambiguity must be exposed, not silently
    resolved (Section 26/36).
    """
    digits_only = normalize_numeric_field(candidate_text)
    m = re.match(r"^(\d{2})(\d{2})(\d{4})$", digits_only)  # DDMMYYYY assumed print order
    if not m:
        m2 = re.match(r"^(\d{4})(\d{2})(\d{2})$", digits_only)  # YYYYMMDD
        if m2:
            y, mo, d = m2.groups()
            return f"{y}-{mo}-{d}"
        return None
    d, mo, y = m.groups()
    return f"{y}-{mo}-{d}"


def parse_gender(candidate_text: str) -> str:
    text = normalize_arabic_text(candidate_text)
    if "ذكر" in text:
        return "male"
    if "أنث" in text or "انث" in text:
        return "female"
    return None
