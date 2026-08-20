"""
Egyptian National ID structural validation (Sections 24-25).

Structure (14 digits):
  [0]      century digit: 2 -> 1900s, 3 -> 2000s
  [1:3]    year within century (YY)
  [3:5]    month (MM)
  [5:7]    day (DD)
  [7:9]    governorate code (birth governorate)
  [9:13]   sequence/serial number for that birth date+governorate;
           the LAST digit of this block ([12]) conventionally encodes
           gender: odd = male, even = female
  [13]     check digit

CAVEAT (do not remove -- Section 1/16/54, "do not hallucinate"):
The Egyptian government has not published an official checksum
specification. The `_compute_check_digit` implementation below follows an
algorithm that is widely circulated in open-source Egyptian-ID validators
(weighted mod-11 over the first 13 digits), but it is NOT independently
confirmed against an authoritative source. Checksum failures are therefore
reported as a *distinct*, lower-trust validation code
(INVALID_CHECKSUM / CHECKSUM_UNVERIFIED-worthy) rather than being fused
into a single pass/fail, so a caller can choose to weight it less than the
structural (date/governorate/length) checks, which ARE derived from
well-established public knowledge of the format.
"""
from dataclasses import dataclass, field
from datetime import date
import calendar

from . import config

VALID = "VALID"
INVALID_FORMAT = "INVALID_FORMAT"
INVALID_DATE = "INVALID_DATE"
INVALID_GOVERNORATE = "INVALID_GOVERNORATE"
INVALID_CHECKSUM = "INVALID_CHECKSUM"
INVALID_STRUCTURE = "INVALID_STRUCTURE"


@dataclass
class NIDValidationResult:
    input_raw: str
    normalized: str
    status: str
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    derived_birth_date: str = None       # ISO date string, if derivable
    derived_governorate_code: str = None
    derived_governorate_name: str = None
    derived_gender: str = None           # "male" | "female"
    checksum_note: str = "Checksum algorithm is unverified against an official spec; treat with caution."

    def as_dict(self):
        return {
            "raw": self.input_raw,
            "normalized": self.normalized,
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            "derived": {
                "birth_date": self.derived_birth_date,
                "governorate_code": self.derived_governorate_code,
                "governorate_name": self.derived_governorate_name,
                "gender": self.derived_gender,
            },
            "checksum_note": self.checksum_note,
        }


def _compute_check_digit(digits13: str) -> int:
    """Best-effort, unverified weighted mod-11 checksum. See module docstring."""
    weights = [2, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
    total = sum(int(d) * w for d, w in zip(digits13, weights))
    remainder = total % 11
    check = 11 - remainder
    if check >= 10:
        check = check % 10
    return check


def validate_nid(raw_or_normalized_digits: str) -> NIDValidationResult:
    from .arabic_normalize import normalize_numeric_field

    normalized = normalize_numeric_field(raw_or_normalized_digits)
    result = NIDValidationResult(input_raw=raw_or_normalized_digits, normalized=normalized, status=VALID)

    if len(normalized) != 14 or not normalized.isdigit():
        result.status = INVALID_FORMAT
        result.errors.append(f"Expected 14 digits, got {len(normalized)} characters.")
        return result

    century_digit = normalized[0]
    yy = normalized[1:3]
    mm = normalized[3:5]
    dd = normalized[5:7]
    gov_code = normalized[7:9]
    serial = normalized[9:13]
    check_digit_str = normalized[13]

    if century_digit not in ("2", "3"):
        result.status = INVALID_STRUCTURE
        result.errors.append(
            f"Unrecognized century digit {century_digit!r} (expected 2=1900s or 3=2000s)."
        )
        return result

    century_base = 1900 if century_digit == "2" else 2000
    try:
        year = century_base + int(yy)
        month = int(mm)
        day = int(dd)
        if not (1 <= month <= 12):
            raise ValueError("month out of range")
        max_day = calendar.monthrange(year, month)[1]
        if not (1 <= day <= max_day):
            raise ValueError("day out of range")
        birth_date = date(year, month, day)
        if birth_date > date.today():
            result.status = INVALID_DATE
            result.errors.append("Derived birth date is in the future.")
            return result
        result.derived_birth_date = birth_date.isoformat()
    except ValueError as e:
        result.status = INVALID_DATE
        result.errors.append(f"Invalid embedded date ({yy}-{mm}-{dd}): {e}")
        return result

    if gov_code not in config.GOVERNORATE_CODES:
        result.status = INVALID_GOVERNORATE
        result.errors.append(f"Unrecognized governorate code {gov_code!r}.")
        return result
    result.derived_governorate_code = gov_code
    result.derived_governorate_name = config.GOVERNORATE_CODES[gov_code]

    if serial == "0000":
        result.warnings.append("Serial block is all zeros; unusual but not necessarily invalid.")

    gender_digit = int(serial[-1])
    result.derived_gender = "male" if gender_digit % 2 == 1 else "female"

    expected_check = _compute_check_digit(normalized[:13])
    if expected_check != int(check_digit_str):
        result.status = INVALID_CHECKSUM
        result.errors.append(
            f"Checksum mismatch: computed {expected_check}, found {check_digit_str}. "
            "NOTE: checksum algorithm is unverified (see checksum_note)."
        )
        return result

    result.status = VALID
    return result
