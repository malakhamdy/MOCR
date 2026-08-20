"""
FastAPI backend for the Egyptian National ID scanner.

This is a thin HTTP layer around the existing, unmodified pipeline in
`core/` (see core/pipeline.py) plus the new, additive `core/forgery_check.py`
heuristics, orchestrated by `core/pipeline_ext.py`.

Flow this API implements:
    Flutter app (auto-captured full-res frame)
      -> POST /api/v1/scan  (multipart image)
      -> core.pipeline_ext.run_full_analysis()
           -> core.pipeline.run_pipeline()          [UNCHANGED existing logic]
           -> core.forgery_check.analyze_document_forgery()  [new, additive]
      -> JSON response: document, fields (with bbox/confidence/status),
         derived, cross_field_validation, forgery_analysis, errors.

No uploaded image is written to disk (mirrors cli.py's existing stance);
everything is handled in memory for the lifetime of the request.
"""
import logging
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core import ocr_engine
from core.pipeline_ext import run_full_analysis, encode_image_base64_jpeg
from . import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("egyptian_id_ocr.api")

app = FastAPI(
    title="Egyptian National ID Scanner API",
    description=(
        "Wraps the existing card-detection / OCR / NID-validation pipeline, "
        "plus experimental forgery-screening heuristics, for the Flutter app."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = settings.MAX_UPLOAD_MB * 1024 * 1024


def _check_api_key(x_api_key: str | None):
    if settings.API_KEY is None:
        return
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "ocr_available": ocr_engine.is_available(),
        "ocr_import_error": None if ocr_engine.is_available() else ocr_engine.import_error(),
    }


@app.post("/api/v1/scan")
async def scan_id(
    request: Request,
    image: UploadFile = File(..., description="JPEG/PNG photo of the ID card (front or back)."),
    run_ocr: bool = Form(settings.DEFAULT_RUN_OCR),
    enable_forgery_check: bool = Form(settings.DEFAULT_ENABLE_FORGERY_CHECK),
    include_canonical_image: bool = Form(settings.DEFAULT_INCLUDE_CANONICAL_IMAGE),
    x_api_key: str | None = Header(default=None),
):
    """
    Main scanning endpoint. Accepts one image (already auto-captured by the
    client), runs the full existing pipeline unchanged, and optionally runs
    the new forgery-screening heuristics.

    Returns the pipeline's document/fields/derived/cross_field_validation
    structure (unchanged shape from core.schema.PipelineResult), plus:
      - forgery_analysis: heuristic screening result (or null if disabled)
      - canonical_card_image_base64: rectified card image (JPEG, base64),
        for the app to overlay the returned bboxes on -- all bboxes are in
        this same canonical-card coordinate space.
    """
    _check_api_key(x_api_key)

    content_length = request.headers.get("content-length")
    if content_length is not None and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {settings.MAX_UPLOAD_MB}MB limit.")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {settings.MAX_UPLOAD_MB}MB limit.")

    np_arr = np.frombuffer(raw, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Send a valid JPEG or PNG.")

    started = time.time()
    try:
        result_dict, canonical_card = run_full_analysis(
            image_bgr, run_ocr=run_ocr, enable_forgery_check=enable_forgery_check
        )
    except Exception as exc:
        logger.exception("Pipeline failure")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {exc}") from exc

    result_dict["canonical_card_image_base64"] = (
        encode_image_base64_jpeg(canonical_card) if (include_canonical_image and canonical_card is not None) else None
    )
    result_dict["processing_time_ms"] = round((time.time() - started) * 1000, 1)

    # jsonable_encoder guards against any stray numpy scalar types slipping
    # through the pipeline's dataclasses (they're cast to native Python
    # types nearly everywhere already, but this makes the encoding robust).
    return JSONResponse(content=jsonable_encoder(result_dict))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
