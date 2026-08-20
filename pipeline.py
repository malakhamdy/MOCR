"""
Pipeline orchestrator (Section 3, 50). Ties every stage together and
returns a schema.PipelineResult with full debug info attached (Section 40).

This module is intentionally defensive: a failure in one field must not
crash the whole run (Section 48). OCR-dependent stages degrade gracefully
if PaddleOCR isn't installed, and still return the geometry/detection/
validation results that don't depend on it (Section 49 -- geometry and
NID math are independently demonstrable even without a live OCR engine).
"""
import numpy as np

from . import config
from .image_utils import normalize_image, assess_image_quality
from .card_detection import detect_and_rectify
from .side_classifier import classify_side
from .templates import get_template_for_side
from .localization import localize_fields
from .preprocessing import preprocess_field
from . import ocr_engine
from .arabic_normalize import normalize_arabic_text, normalize_numeric_field
from . import field_parsers
from .ocr_postprocess import merge_field_candidates, rank_field_candidates
from .nid_validator import validate_nid
from .cross_field_validator import run_cross_field_validation
from . import independent_verifier
from .barcode import decode_barcode
from .schema import DocumentResult, FieldResult, PipelineResult


def run_pipeline(image_bgr: np.ndarray, run_ocr: bool = True) -> PipelineResult:
    debug = {}
    errors = []

    # --- 1. quality gate on raw upload -----------------------------------
    raw_quality = assess_image_quality(image_bgr)
    debug["raw_image_quality"] = raw_quality

    # --- 2. normalize (Section 7) -----------------------------------------
    processing_img, transform, original_img = normalize_image(image_bgr)
    debug["normalization_transform"] = transform.as_dict()

    # --- 3. card detection + rectification (Sections 9-10) -----------------
    detection = detect_and_rectify(processing_img)
    debug["card_detection"] = {
        "corners": detection["corners"],
        "confidence": detection["confidence"],
        "debug": detection["debug"],
    }

    document = DocumentResult(
        is_egyptian_id=False,
        card_detection_confidence=detection["confidence"],
        image_quality=raw_quality,
    )

    if detection["canonical_card"] is None:
        errors.append("Card not detected: no plausible quadrilateral found in the image.")
        return PipelineResult(document=document, fields={}, debug=debug, errors=errors)

    canonical_card = detection["canonical_card"]
    document.is_egyptian_id = True  # detected *a* card; content validation happens later

    # --- 4. front/back classification (Section 14) --------------------------
    side_info = classify_side(canonical_card)
    document.side = side_info["side"]
    document.side_confidence = side_info["confidence"]
    debug["side_classification"] = side_info

    if document.side == config.SIDE_UNKNOWN:
        errors.append("Could not confidently classify front vs back; skipping field localization.")
        debug["canonical_card_available"] = True
        return PipelineResult(document=document, fields={}, debug=debug, errors=errors)

    template = get_template_for_side(document.side)
    document.template = template.template_id

    # --- 5. dynamic field localization + crops (Sections 16, 19) ------------
    localized = localize_fields(canonical_card, template)
    debug["localization"] = {
        name: {
            "bbox": info["bbox"].as_dict() if info["bbox"] else None,
            "crop_quality": info["crop_quality"],
            "localization_confidence": info["localization_confidence"],
        }
        for name, info in localized.items()
    }

    fields = {}
    ocr_candidates_debug = {}

    ocr_ready = run_ocr and ocr_engine.is_available()
    if run_ocr and not ocr_engine.is_available():
        errors.append(
            "OCR requested but PaddleOCR is not installed in this environment "
            f"({ocr_engine.import_error()}); returning geometry/localization only."
        )

    for field_name, info in localized.items():
        crop = info["crop"]
        field_type = info["field_type"]
        fr = FieldResult(
            field=field_name,
            bbox=info["bbox"].as_dict() if info["bbox"] else None,
            coordinate_space=info["bbox"].coordinate_space if info["bbox"] else None,
            localization_confidence=info["localization_confidence"],
            status=config.STATUS_DETECTED,
        )

        if not info["crop_quality"]["acceptable"]:
            fr.issues.extend(info["crop_quality"]["issues"])

        if crop is None or crop.size == 0:
            fr.status = config.STATUS_FAILED
            fr.issues.append("empty_crop")
            fields[field_name] = fr
            continue

        if field_type == "barcode":
            bc = decode_barcode(crop)
            fr.raw = bc.get("payload")
            fr.normalized = bc.get("payload")
            fr.status = config.STATUS_EXTRACTED if bc["status"] == "barcode_decoded" else config.STATUS_DETECTED
            fr.validation = bc
            fields[field_name] = fr
            continue

        if field_type == "image_region":
            fr.status = config.STATUS_DETECTED
            fr.issues.append("not_an_ocr_field")
            fields[field_name] = fr
            continue

        variants = preprocess_field(field_name, crop, field_type)

        if not ocr_ready:
            fr.status = config.STATUS_FAILED
            fr.issues.append("ocr_unavailable")
            fields[field_name] = fr
            continue

        variant_results = ocr_engine.run_ocr_multi_variant(variants, lang="ar")
        ocr_candidates_debug[field_name] = {
            v: [c.__dict__ for c in cands] for v, cands in variant_results.items()
        }

        all_candidates = [c for cands in variant_results.values() for c in cands]
        merged_candidates = merge_field_candidates(all_candidates, field_name)
        ranked_candidates = rank_field_candidates(merged_candidates, field_name)
        ocr_candidates_debug[field_name + "_merged"] = ranked_candidates

        if not ranked_candidates:
            fr.status = config.STATUS_FAILED
            fr.issues.append("no_text_detected")
            fields[field_name] = fr
            continue

        best = ranked_candidates[0]
        is_numeric = field_type == "numeric"
        fr.raw = best["text"]
        fr.normalized = (normalize_numeric_field(best["text"]) if is_numeric
                          else normalize_arabic_text(best["text"]))
        fr.engine = "paddleocr"
        fr.preprocessing_variant = best["variant"]
        fr.ocr_confidence = best["confidence"]
        fr.sources = ["printed_ocr"]
        fr.status = (config.STATUS_LOW_CONFIDENCE if best["confidence"] < config.LOW_CONFIDENCE_OCR
                     else config.STATUS_EXTRACTED)

        # If two top candidates disagree materially, expose that disagreement
        # instead of silently treating a fragile OCR read as certain.
        if len(ranked_candidates) > 1 and ranked_candidates[0]["normalized"] != ranked_candidates[1]["normalized"]:
            fr.issues.append("multiple_ocr_candidates_disagree")

        fields[field_name] = fr

    debug["ocr_candidates"] = ocr_candidates_debug

    # --- 6. NID validation (Sections 24-25) ---------------------------------
    derived = {}
    nid_field = fields.get("national_id")
    nid_validation_dict = None
    if nid_field and nid_field.normalized:
        nid_result = validate_nid(nid_field.normalized)
        nid_validation_dict = nid_result.as_dict()
        nid_field.validation = nid_validation_dict
        nid_field.verification = independent_verifier.verify_national_id(nid_validation_dict)
        nid_field.status = nid_field.verification["status"]
        derived["birth_governorate"] = {
            "code": nid_validation_dict["derived"]["governorate_code"],
            "name": nid_validation_dict["derived"]["governorate_name"],
            "source": "nid_structure",
        }
        derived["gender"] = {
            "value": nid_validation_dict["derived"]["gender"],
            "source": "nid_structure",
        }
        derived["date_of_birth"] = {
            "value": nid_validation_dict["derived"]["birth_date"],
            "source": "nid_structure",
        }
        # Backwards-compatible aliases used by earlier callers.
        derived["nid_derived_dob"] = nid_validation_dict["derived"]["birth_date"]

    # --- 7. cross-field validation (Section 35) ------------------------------
    cross_extracted = {}
    if "date_of_birth" in fields and fields["date_of_birth"].raw:
        parsed_dob = field_parsers.parse_dob_printed(fields["date_of_birth"].raw)
        cross_extracted["date_of_birth_printed"] = parsed_dob
        if nid_validation_dict:
            dob_verify = independent_verifier.verify_dob(
                parsed_dob, nid_validation_dict["derived"]["birth_date"]
            )
            fields["date_of_birth"].verification = dob_verify
            fields["date_of_birth"].status = dob_verify["status"]

    if "gender" in fields and fields["gender"].raw:
        parsed_gender = field_parsers.parse_gender(fields["gender"].raw)
        cross_extracted["gender_printed"] = parsed_gender
        if nid_validation_dict:
            gender_verify = independent_verifier.verify_gender(
                parsed_gender, nid_validation_dict["derived"]["gender"]
            )
            fields["gender"].verification = gender_verify
            fields["gender"].status = gender_verify["status"]

    cross_field_result = {}
    if nid_validation_dict:
        cross_field_result = run_cross_field_validation(cross_extracted, nid_validation_dict)

    return PipelineResult(
        document=document,
        fields=fields,
        derived=derived,
        cross_field_validation=cross_field_result,
        debug=debug,
        errors=errors,
    )
