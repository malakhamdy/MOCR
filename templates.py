"""
Template registry (Section 15).

Field regions are stored as NORMALIZED (fraction-of-card) coordinates so
the same template works regardless of input image resolution -- they are
converted to canonical-card pixel coordinates at runtime (Section 12/16).

NOTE ON ACCURACY: the exact region fractions below are a best-effort
starting layout based on the publicly known general structure of the
current-issue Egyptian National ID (front: photo + name/DOB/address block
on the right for RTL layout, NID number bottom strip; back: job/marital
fields + barcode strip). These fractions are almost certainly not pixel
perfect for every physical print run and MUST be recalibrated against real
sample cards (Section 53) -- that recalibration is exactly what
localization.py's semantic-anchor refinement step is for. Do not treat
these numbers as ground truth; treat them as a coarse prior that anchors
detection.
"""
from dataclasses import dataclass, field as dc_field
from .coordinate_systems import NormalizedBBox
from . import config


@dataclass
class FieldSpec:
    name: str
    region: NormalizedBBox
    field_type: str  # "arabic_text" | "numeric" | "arabic_numeric" | "barcode" | "image_region"
    required: bool = True


@dataclass
class Template:
    template_id: str
    side: str
    fields: list


FRONT_TEMPLATE_V1 = Template(
    template_id="egy_id_front_v1",
    side=config.SIDE_FRONT,
    fields=[
        FieldSpec("photo", NormalizedBBox(0.02, 0.08, 0.24, 0.52), "image_region"),
        # Recalibrated 2026-08-19 against two real sample cards (measured ink-band
        # positions on the canonical card -- see git history / conversation for the
        # measurement script). Prior version of this template had "name" and
        # "address" swapped with the card's title header; that was caught only by
        # testing on real samples, not guessed correctly upfront. Regions are
        # padded generously because the current card_detection full-frame
        # fallback (used when no clean background border exists for contour
        # detection) does not yet do true corner-precise rectification, so
        # vertical alignment still shifts a bit card-to-card -- tighten these
        # once corner-accurate detection is validated on more samples.
        FieldSpec("name", NormalizedBBox(0.28, 0.28, 0.70, 0.18), "arabic_text"),
        FieldSpec("address", NormalizedBBox(0.28, 0.43, 0.70, 0.25), "arabic_text"),
        FieldSpec("national_id", NormalizedBBox(0.28, 0.74, 0.70, 0.16), "numeric"),
    ],
)

BACK_TEMPLATE_V1 = Template(
    template_id="egy_id_back_v1",
    side=config.SIDE_BACK,
    fields=[
        FieldSpec("job_or_status", NormalizedBBox(0.05, 0.06, 0.55, 0.14), "arabic_text"),
        FieldSpec("gender", NormalizedBBox(0.05, 0.22, 0.30, 0.12), "arabic_text"),
        FieldSpec("religion", NormalizedBBox(0.38, 0.22, 0.30, 0.12), "arabic_text", required=False),
        FieldSpec("marital_status", NormalizedBBox(0.05, 0.36, 0.30, 0.12), "arabic_text", required=False),
        FieldSpec("husband_name", NormalizedBBox(0.05, 0.50, 0.60, 0.12), "arabic_text", required=False),
        FieldSpec("barcode", NormalizedBBox(0.55, 0.60, 0.42, 0.35), "barcode"),
    ],
)

TEMPLATE_REGISTRY = {
    FRONT_TEMPLATE_V1.template_id: FRONT_TEMPLATE_V1,
    BACK_TEMPLATE_V1.template_id: BACK_TEMPLATE_V1,
}


def get_template_for_side(side: str) -> Template:
    if side == config.SIDE_FRONT:
        return FRONT_TEMPLATE_V1
    if side == config.SIDE_BACK:
        return BACK_TEMPLATE_V1
    raise ValueError(f"No template registered for side={side!r}")
