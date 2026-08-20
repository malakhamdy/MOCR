"""PaddleOCR adapter with compatibility for PaddleOCR 2.x and 3.x.

The project is field-first: localization happens before OCR, so each crop is
sent through several preprocessing variants.  PaddleOCR 3.x uses ``predict``
and returns ``rec_texts``/``rec_scores``/``rec_polys``; 2.x uses ``ocr`` and
returns ``[[poly, (text, score)], ...]``.  This adapter normalizes both APIs to
one OCRCandidate representation.
"""
from dataclasses import dataclass
import inspect
import numpy as np
import cv2

_PADDLE_IMPORT_ERROR = None
try:
    from paddleocr import PaddleOCR
    _PADDLE_AVAILABLE = True
except Exception as e:  # pragma: no cover
    _PADDLE_AVAILABLE = False
    _PADDLE_IMPORT_ERROR = e


class OCRUnavailableError(RuntimeError):
    pass


@dataclass
class OCRCandidate:
    text: str
    confidence: float
    bbox: list
    engine: str
    variant: str


_ENGINE_CACHE = {}


def is_available() -> bool:
    return _PADDLE_AVAILABLE


def import_error() -> str:
    return str(_PADDLE_IMPORT_ERROR) if _PADDLE_IMPORT_ERROR else ""


def _detect_gpu() -> bool:
    try:
        import paddle
        return bool(paddle.device.cuda.device_count())
    except Exception:
        return False


def _supports_kwarg(callable_obj, name: str) -> bool:
    try:
        return name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


def get_engine(lang: str = "ar", use_gpu: bool = None):
    """Return one cached PaddleOCR instance per language/device."""
    if not _PADDLE_AVAILABLE:
        raise OCRUnavailableError(
            "PaddleOCR is not installed. "
            f"Import error: {_PADDLE_IMPORT_ERROR}."
        )

    if use_gpu is None:
        use_gpu = _detect_gpu()
    cache_key = (lang, bool(use_gpu))
    if cache_key in _ENGINE_CACHE:
        return _ENGINE_CACHE[cache_key]

    sig = inspect.signature(PaddleOCR)
    params = sig.parameters
    kwargs = {}

    # PaddleOCR 3.x: disable document-level transforms because the card has
    # already been rectified and each field crop has a known orientation.
    if "use_doc_orientation_classify" in params:
        kwargs["use_doc_orientation_classify"] = False
    if "use_doc_unwarping" in params:
        kwargs["use_doc_unwarping"] = False
    if "use_textline_orientation" in params:
        kwargs["use_textline_orientation"] = False
    if "lang" in params:
        kwargs["lang"] = lang
    if "engine" in params:
        kwargs["engine"] = "paddle"
    if "use_gpu" in params and not "device" in params:
        kwargs["use_gpu"] = bool(use_gpu)
    if "device" in params:
        kwargs["device"] = "gpu:0" if use_gpu else "cpu"

    # PaddleOCR 2.x accepts use_angle_cls and optionally ocr_version.
    if "use_angle_cls" in params:
        kwargs["use_angle_cls"] = True
    if "ocr_version" in params:
        kwargs["ocr_version"] = "PP-OCRv5"

    try:
        engine = PaddleOCR(**kwargs)
    except Exception as first_error:
        # A conservative compatibility retry for installations where one of
        # the optional parameters is accepted by the signature but rejected by
        # the installed backend/model package.
        retry = {k: v for k, v in kwargs.items()
                 if k not in {"ocr_version", "engine", "device", "use_gpu"}}
        try:
            engine = PaddleOCR(**retry)
        except Exception as second_error:
            raise OCRUnavailableError(
                f"Could not initialize PaddleOCR. First error: {first_error}; "
                f"retry error: {second_error}"
            ) from second_error

    _ENGINE_CACHE[cache_key] = engine
    return engine


def _result_to_dict(result):
    """Best-effort conversion of a PaddleOCR 3.x result object to a dict."""
    for attr in ("json", "to_dict"):
        try:
            value = getattr(result, attr)
            value = value() if callable(value) else value
            if isinstance(value, dict):
                return value.get("res", value)
            if isinstance(value, str):
                import json
                parsed = json.loads(value)
                return parsed.get("res", parsed)
        except Exception:
            pass
    if isinstance(result, dict):
        return result.get("res", result)
    return None


def _run_legacy(engine, image_bgr):
    raw = engine.ocr(image_bgr, cls=True)
    if not raw or raw[0] is None:
        return []
    out = []
    for line in raw[0]:
        if not line or len(line) < 2:
            continue
        poly, rec = line
        if not rec or len(rec) < 2:
            continue
        text, conf = rec
        if text is None:
            continue
        out.append((text, float(conf), np.asarray(poly).tolist()))
    return out


def _run_v3(engine, image):
    out = []
    for result in engine.predict(image):
        data = _result_to_dict(result)
        if not data:
            continue
        texts = data.get("rec_texts") or []
        scores = data.get("rec_scores") or []
        polys = data.get("rec_polys")
        if polys is None:
            polys = data.get("dt_polys")
        if polys is None:
            polys = []
        boxes = data.get("rec_boxes")
        for i, text in enumerate(texts):
            if text is None:
                continue
            score = float(scores[i]) if i < len(scores) else 0.0
            if i < len(polys):
                poly = np.asarray(polys[i]).tolist()
            elif boxes is not None and i < len(boxes):
                x1, y1, x2, y2 = [int(v) for v in boxes[i]]
                poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            else:
                poly = []
            out.append((str(text), score, poly))
    return out


def run_ocr(image: np.ndarray, lang: str = "ar", variant: str = "default") -> list:
    if not _PADDLE_AVAILABLE:
        raise OCRUnavailableError(
            "PaddleOCR is not installed; cannot run OCR. "
            f"Import error: {_PADDLE_IMPORT_ERROR}"
        )
    if image is None or image.size == 0:
        return []

    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    engine = get_engine(lang=lang)

    try:
        if hasattr(engine, "predict"):
            raw_candidates = _run_v3(engine, image_bgr)
        else:
            raw_candidates = _run_legacy(engine, image_bgr)
    except Exception as e:
        raise OCRUnavailableError(f"PaddleOCR inference failed: {e}") from e

    return [OCRCandidate(
        text=text,
        confidence=max(0.0, min(float(conf), 1.0)),
        bbox=poly,
        engine="paddleocr",
        variant=variant,
    ) for text, conf, poly in raw_candidates]


def run_ocr_multi_variant(variants: dict, lang: str = "ar") -> dict:
    results = {}
    for name, img in variants.items():
        try:
            results[name] = run_ocr(img, lang=lang, variant=name)
        except OCRUnavailableError:
            results[name] = []
    return results
