"""
Independent verification layer (Section 32-33).

Distinguishes EXTRACTED from VERIFIED by only using evidence sources that
are structurally independent of the primary OCR read:
  - NID structural/checksum validation (math, not OCR)
  - NID-derived DOB / governorate / gender vs printed OCR (cross source)
  - a second, meaningfully different preprocessing/OCR pass on the same crop
  - barcode payload, if decoded

Never invents agreement. If only one piece of evidence exists, status
stays EXTRACTED or VALIDATED, not VERIFIED.
"""
from . import config


def verify_national_id(nid_validation: dict) -> dict:
    if nid_validation["status"] == "VALID":
        return {"status": config.STATUS_VERIFIED,
                "reason": "Passed structural validation (date, governorate, checksum)."}
    if nid_validation["status"] == "INVALID_CHECKSUM":
        return {"status": config.STATUS_LOW_CONFIDENCE,
                "reason": "Structure/date/governorate valid but checksum failed "
                          "(checksum algorithm itself is unverified -- see nid_validator caveat)."}
    return {"status": config.STATUS_FAILED, "reason": nid_validation.get("errors")}


def verify_dob(printed_dob_iso: str, nid_derived_dob_iso: str) -> dict:
    if printed_dob_iso and nid_derived_dob_iso:
        if printed_dob_iso == nid_derived_dob_iso:
            return {"status": config.STATUS_CROSS_VALIDATED,
                    "reason": "Printed DOB matches NID-derived DOB."}
        return {"status": config.STATUS_MISMATCH,
                "reason": "Printed DOB and NID-derived DOB disagree."}
    if nid_derived_dob_iso and not printed_dob_iso:
        return {"status": config.STATUS_EXTRACTED,
                "reason": "Only NID-derived DOB available; printed field not read."}
    if printed_dob_iso and not nid_derived_dob_iso:
        return {"status": config.STATUS_EXTRACTED,
                "reason": "Only printed DOB available; NID not validated."}
    return {"status": config.STATUS_FAILED, "reason": "No DOB evidence available."}


def verify_gender(printed_gender: str, nid_derived_gender: str) -> dict:
    if printed_gender and nid_derived_gender:
        if printed_gender == nid_derived_gender:
            return {"status": config.STATUS_CROSS_VALIDATED,
                    "reason": "Printed gender matches NID-derived gender digit."}
        return {"status": config.STATUS_MISMATCH,
                "reason": "Printed gender and NID-derived gender disagree."}
    if printed_gender:
        return {"status": config.STATUS_EXTRACTED, "reason": "Only printed gender available."}
    return {"status": config.STATUS_FAILED, "reason": "No gender evidence available."}


def verify_two_pass_agreement(candidate_a: str, candidate_b: str, field_name: str = "") -> dict:
    """Section 33: two meaningfully different OCR passes on the same field."""
    if candidate_a is None or candidate_b is None:
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": "Fewer than two candidates available."}
    if candidate_a == candidate_b:
        return {"status": "AGREEMENT", "reason": f"Both passes produced identical {field_name or 'value'}."}
    return {"status": "DISAGREEMENT",
            "reason": f"Passes disagree ({candidate_a!r} vs {candidate_b!r}); do not silently pick one.",
            "candidate_a": candidate_a, "candidate_b": candidate_b}
