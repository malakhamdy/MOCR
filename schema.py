"""
Structured result schema with full provenance (Section 34, 47).
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from . import config


@dataclass
class FieldResult:
    field: str
    raw: str = None
    normalized: str = None
    sources: list = field(default_factory=list)   # e.g. ["printed_ocr"], ["printed_ocr","nid_derived"]
    engine: str = None
    preprocessing_variant: str = None
    bbox: dict = None
    coordinate_space: str = None
    ocr_confidence: float = None
    localization_confidence: float = None
    validation: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    status: str = config.STATUS_DETECTED
    issues: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


@dataclass
class DocumentResult:
    is_egyptian_id: bool = False
    side: str = config.SIDE_UNKNOWN
    side_confidence: float = 0.0
    card_detection_confidence: float = 0.0
    template: str = None
    image_quality: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


@dataclass
class PipelineResult:
    document: DocumentResult
    fields: dict            # {field_name: FieldResult}
    derived: dict = field(default_factory=dict)
    cross_field_validation: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    def as_dict(self):
        return {
            "document": self.document.as_dict(),
            "fields": {k: v.as_dict() if isinstance(v, FieldResult) else v for k, v in self.fields.items()},
            "derived": self.derived,
            "cross_field_validation": self.cross_field_validation,
            "debug": self.debug,
            "errors": self.errors,
        }
