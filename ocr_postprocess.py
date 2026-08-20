"""Field-aware OCR candidate merging and ranking.

OCR engines often return several boxes for one field.  Picking the single
highest-confidence box is a major failure mode for ID cards: a name may be
split into multiple words and a 14-digit number may be split into chunks.
This module merges boxes spatially, then ranks the resulting candidates using
field-specific constraints without inventing characters.
"""
import re
import numpy as np
from .arabic_normalize import normalize_arabic_text, normalize_numeric_field
from .nid_validator import validate_nid


def _bbox_center(poly):
    if not poly:
        return 0.0, 0.0
    a = np.asarray(poly, dtype=float).reshape(-1, 2)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


def _line_groups(candidates, y_tol_ratio=0.65):
    items = []
    for c in candidates:
        if not c.text or not c.text.strip():
            continue
        x, y = _bbox_center(c.bbox)
        a = np.asarray(c.bbox, dtype=float).reshape(-1, 2) if c.bbox else np.zeros((4, 2))
        height = float(max(a[:, 1]) - min(a[:, 1])) if len(a) else 20.0
        items.append((c, x, y, max(height, 1.0)))

    items.sort(key=lambda t: t[2])
    groups = []
    for item in items:
        _, _, y, h = item
        placed = False
        for group in groups:
            gy = np.mean([g[2] for g in group])
            gh = np.mean([g[3] for g in group])
            if abs(y - gy) <= max(gh, h) * y_tol_ratio:
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])
    return groups


def merge_field_candidates(candidates, field_name):
    """Return candidate strings built from all detected text boxes."""
    if not candidates:
        return []

    groups = _line_groups(candidates)
    merged = []
    for group in groups:
        # Arabic/RTL fields: rightmost box first. Numeric fields are normally
        # left-to-right as printed digits, so preserve left-to-right order.
        reverse = field_name not in {"national_id", "date_of_birth"}
        ordered = sorted(group, key=lambda t: t[1], reverse=reverse)
        text = " ".join(t[0].text.strip() for t in ordered if t[0].text.strip())
        conf = float(np.mean([t[0].confidence for t in ordered]))
        variants = sorted({t[0].variant for t in ordered})
        merged.append({
            "text": text,
            "confidence": conf,
            "variant": "+".join(variants),
            "box_count": len(ordered),
        })

    # Also keep a whole-field concatenation. This is especially useful for a
    # 14-digit NID detected as multiple adjacent chunks.
    if field_name in {"national_id", "date_of_birth"} and len(groups) >= 1:
        ordered_groups = sorted(groups, key=lambda g: np.mean([t[2] for t in g]))
        chunks = []
        confs = []
        variants = set()
        for group in ordered_groups:
            ordered = sorted(group, key=lambda t: t[1])
            chunks.extend(t[0].text.strip() for t in ordered if t[0].text.strip())
            confs.extend(t[0].confidence for t in ordered)
            variants.update(t[0].variant for t in ordered)
        if chunks:
            merged.append({
                "text": "".join(chunks),
                "confidence": float(np.mean(confs)),
                "variant": "+".join(sorted(variants)),
                "box_count": len(chunks),
            })
    return merged


def rank_field_candidates(candidates, field_name):
    """Return candidates sorted best-first; no character correction is done."""
    ranked = []
    for c in candidates:
        raw = c["text"]
        if field_name == "national_id":
            normalized = normalize_numeric_field(raw)
            length_score = 1.0 if len(normalized) == 14 else max(0.0, 1.0 - abs(14 - len(normalized)) / 14)
            validation_bonus = 0.0
            validation_status = None
            if len(normalized) == 14:
                result = validate_nid(normalized)
                validation_status = result.status
                if result.status == "VALID":
                    validation_bonus = 0.35
                elif result.status in {"INVALID_CHECKSUM"}:
                    validation_bonus = 0.12
                elif not result.errors:
                    validation_bonus = 0.05
            score = 0.55 * c["confidence"] + 0.30 * length_score + 0.15 * min(validation_bonus / 0.35, 1.0)
            ranked.append({**c, "normalized": normalized, "validation_status": validation_status, "rank_score": score})
        else:
            normalized = normalize_arabic_text(raw)
            nonempty = 1.0 if normalized else 0.0
            score = 0.75 * c["confidence"] + 0.25 * nonempty
            ranked.append({**c, "normalized": normalized, "rank_score": score})
    return sorted(ranked, key=lambda x: x["rank_score"], reverse=True)
