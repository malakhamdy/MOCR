"""
Explicit coordinate-space bookkeeping (Section 8).

Every bbox that moves through the pipeline should be wrapped in a BBox so
its coordinate space travels with it and spaces are never silently mixed.
"""
from dataclasses import dataclass, field
from . import config


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float
    coordinate_space: str  # one of config.SPACE_*

    def as_dict(self):
        return {
            "bbox": [self.x, self.y, self.w, self.h],
            "coordinate_space": self.coordinate_space,
        }

    def to_normalized(self, ref_w: float, ref_h: float) -> "NormalizedBBox":
        return NormalizedBBox(
            x=self.x / ref_w, y=self.y / ref_h,
            w=self.w / ref_w, h=self.h / ref_h,
        )


@dataclass
class NormalizedBBox:
    """Fraction-of-card coordinates (Section 12). This is the portable,
    resolution-independent representation used by the template registry."""
    x: float
    y: float
    w: float
    h: float

    def to_canonical(self, card_w: float = config.CANONICAL_CARD_WIDTH,
                      card_h: float = config.CANONICAL_CARD_HEIGHT) -> BBox:
        return BBox(
            x=self.x * card_w, y=self.y * card_h,
            w=self.w * card_w, h=self.h * card_h,
            coordinate_space=config.SPACE_CANONICAL,
        )
