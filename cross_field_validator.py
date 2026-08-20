"""
Cross-field consistency engine (Section 35).
"""
from . import config


def check_dob_consistency(printed_dob_iso: str, nid_derived_dob_iso: str) -> dict:
    if not printed_dob_iso or not nid_derived_dob_iso:
        return {"check": "dob_consistency", "result": "INSUFFICIENT_EVIDENCE"}
    if printed_dob_iso == nid_derived_dob_iso:
        return {"check": "dob_consistency", "result": "MATCH",
                "printed": printed_dob_iso, "nid_derived": nid_derived_dob_iso}
    return {"check": "dob_consistency", "result": "MISMATCH",
            "printed": printed_dob_iso, "nid_derived": nid_derived_dob_iso}


def check_gender_consistency(printed_gender: str, nid_derived_gender: str) -> dict:
    if not printed_gender or not nid_derived_gender:
        return {"check": "gender_consistency", "result": "INSUFFICIENT_EVIDENCE"}
    if printed_gender == nid_derived_gender:
        return {"check": "gender_consistency", "result": "MATCH",
                "printed": printed_gender, "nid_derived": nid_derived_gender}
    return {"check": "gender_consistency", "result": "MISMATCH",
            "printed": printed_gender, "nid_derived": nid_derived_gender}


def check_governorate_consistency(printed_governorate_name: str, nid_derived_governorate_name: str) -> dict:
    if not printed_governorate_name or not nid_derived_governorate_name:
        return {"check": "governorate_consistency", "result": "INSUFFICIENT_EVIDENCE"}
    match = printed_governorate_name.strip() == nid_derived_governorate_name.strip()
    return {
        "check": "governorate_consistency",
        "result": "MATCH" if match else "MISMATCH",
        "printed": printed_governorate_name,
        "nid_derived": nid_derived_governorate_name,
        "note": "NID code reflects BIRTH governorate, not necessarily current residence.",
    }


def run_cross_field_validation(extracted: dict, nid_validation: dict) -> dict:
    """
    extracted: dict of parsed field values, e.g.
        {"date_of_birth_printed": "1999-04-17", "gender_printed": "male", ...}
    nid_validation: NIDValidationResult.as_dict()
    """
    checks = []
    checks.append(check_dob_consistency(
        extracted.get("date_of_birth_printed"),
        nid_validation.get("derived", {}).get("birth_date"),
    ))
    checks.append(check_gender_consistency(
        extracted.get("gender_printed"),
        nid_validation.get("derived", {}).get("gender"),
    ))
    checks.append(check_governorate_consistency(
        extracted.get("birth_governorate_printed"),
        nid_validation.get("derived", {}).get("governorate_name"),
    ))

    matches = [c for c in checks if c["result"] == "MATCH"]
    mismatches = [c for c in checks if c["result"] == "MISMATCH"]
    warnings = [c for c in checks if c["result"] == "INSUFFICIENT_EVIDENCE"]

    return {
        "checks": checks,
        "matches": matches,
        "mismatches": mismatches,
        "warnings": warnings,
        "overall_consistent": len(mismatches) == 0,
    }
