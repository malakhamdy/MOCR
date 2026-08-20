# Egyptian National ID Scanner — FastAPI backend

This is a thin HTTP layer around the **existing, unmodified** pipeline in
`core/`. Nothing in `core/pipeline.py`, `core/card_detection.py`,
`core/localization.py`, `core/ocr_engine.py`, `core/nid_validator.py`, or
`core/cross_field_validator.py` was changed.

Two new files were added on top, and nothing else was touched:

- `core/forgery_check.py` — new, additive, clearly-experimental forgery
  screening heuristics (see disclaimer below and in the module docstring).
- `core/pipeline_ext.py` — orchestrates the unmodified `run_pipeline()`
  plus the new forgery checks, for the API to call.
- `api/` — the FastAPI app itself (`main.py`, `settings.py`).

## Why no YOLO

The project as delivered had a Streamlit UI toggle labeled "YOLO
Segmentation (AI Model)" and a `yolov8n-seg.pt` file, but no code wiring it
into `core/`, and that `.pt` file is a generic COCO-pretrained
segmentation checkpoint — not trained on ID cards. Wiring it in as-is
would silently produce meaningless field boxes. Per your choice, this
backend uses the real, working localization path only: classical CV card
detection (`card_detection.py`) + dynamic anchor-based field localization
(`localization.py`), same as the existing pipeline already used. If you
later have an ID-card-trained YOLO model, `core/localization.py` is the
natural place to add it as an alternate localizer — happy to wire that in
when you have real weights.

## Running locally

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Or with Docker:

```bash
docker build -t egyptian-id-ocr-api .
docker run -p 8000:8000 egyptian-id-ocr-api
```

Interactive API docs: `http://localhost:8000/docs`

Note: `paddleocr`/`paddlepaddle` are heavy native dependencies. If they
aren't installed, `/api/v1/health` will report `ocr_available: false` and
`/api/v1/scan` will still return card detection, localization, geometry,
and forgery-screening results, but OCR'd field text will come back with
`status: "FAILED"` / `issue: "ocr_unavailable"` — exactly the same
graceful-degradation behavior `core/pipeline.py` already had before this
integration.

## Endpoints

### `GET /api/v1/health`

```json
{"status": "ok", "ocr_available": true, "ocr_import_error": null}
```

### `POST /api/v1/scan`

`multipart/form-data`:

| field                     | type | default | notes                                              |
|---------------------------|------|---------|-----------------------------------------------------|
| `image`                   | file | —       | required. JPEG/PNG, front or back of the ID.        |
| `run_ocr`                 | bool | `true`  | set `false` for geometry/detection-only (fast).     |
| `enable_forgery_check`    | bool | `true`  | runs the new heuristic screening (see below).       |
| `include_canonical_image` | bool | `true`  | include the rectified card as base64 JPEG.          |

Optional header: `X-API-Key` — only enforced if the server has
`EID_API_KEY` set in its environment (unset by default, i.e. no auth,
which is fine for local dev only).

Response shape (unchanged core fields + 3 additions at the end):

```jsonc
{
  "document": {
    "is_egyptian_id": true,
    "side": "front",
    "side_confidence": 0.98,
    "card_detection_confidence": 0.82,
    "template": "egy_id_front_v1",
    "image_quality": { "blur_score": 1571.4, "brightness": 226.7, "glare_ratio": 0.38, "issues": ["possible_glare"], "acceptable": false }
  },
  "fields": {
    "national_id": {
      "field": "national_id",
      "raw": "29001011234567",
      "normalized": "29001011234567",
      "bbox": [x, y, w, h],
      "coordinate_space": "canonical_card",
      "ocr_confidence": 0.91,
      "localization_confidence": 0.9,
      "validation": { "...": "NID checksum / structure result" },
      "verification": { "status": "VERIFIED", "...": "..." },
      "status": "VERIFIED",
      "issues": []
    },
    "name": { "...": "..." },
    "address": { "...": "..." },
    "photo": { "field": "photo", "status": "DETECTED", "issues": ["not_an_ocr_field"], "bbox": [...] }
  },
  "derived": { "birth_governorate": {...}, "gender": {...}, "date_of_birth": {...} },
  "cross_field_validation": { "...": "..." },
  "debug": { "...": "card detection / localization / OCR-candidate debug info" },
  "errors": [],

  // --- new, additive fields ---
  "forgery_analysis": {
    "status": "SUSPICIOUS",
    "suspicion_score": 0.52,
    "checks": {
      "error_level_analysis": { "ran": true, "score": 0.1, "flagged": false, "detail": "..." },
      "copy_move_detection": { "ran": true, "score": 0.4, "flagged": true, "detail": "...", "reason": "..." },
      "screen_recapture_check": { "...": "..." },
      "font_consistency_check": { "...": "..." },
      "photo_boundary_check": { "...": "..." }
    },
    "reasons": ["Found repeated, spatially-distant patches that may indicate a pasted/cloned region."],
    "disclaimer": "Experimental, unsupervised heuristics... not a certified forensic tool..."
  },
  "canonical_card_image_base64": "<jpeg bytes, base64>",
  "processing_time_ms": 623.8
}
```

All `bbox` values are in `"coordinate_space": "canonical_card"` — i.e. pixel
coordinates on the rectified card image at
`config.CANONICAL_CARD_WIDTH × CANONICAL_CARD_HEIGHT` (1600×1009). Draw
overlays against `canonical_card_image_base64`, not the original photo.

## `forgery_analysis` — please read this before wiring it into any UX

This is **new code that did not exist in the project you gave me**. It is
5 classical, unsupervised image-forensics heuristics (Error Level
Analysis, ORB-based copy-move detection, an FFT-based screen/moire check,
stroke-width consistency across text fields, and a photo-edge-halo check).
None of them are trained on labeled real/fake Egyptian ID samples, and
none of them are a substitute for a real forgery-detection model or human
review. Expect false positives (e.g. `SUSPICIOUS` on a genuine but
JPEG-recompressed or oddly-lit photo) and false negatives (a competent
edit can dodge all five checks). Treat `forgery_analysis.status` as "maybe
route this one to a human," never as an automatic reject. This is stated
in the response's own `disclaimer` field so it's visible to every
consumer of the API, not just readers of this doc.

## What's next (Flutter app)

The Flutter Android app (live camera preview, on-device auto card
detection / blur-glare check / auto-capture using Dart-only heuristics, per
your earlier choice) will POST the auto-captured full-resolution frame to
`POST /api/v1/scan` and render `fields`, `document`, `forgery_analysis`
using the bboxes against `canonical_card_image_base64`. That's the next
piece I'll build.
