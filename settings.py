import os


def _env_list(name: str, default: list) -> list:
    raw = os.environ.get(name)
    if not raw:
        return default
    return [v.strip() for v in raw.split(",") if v.strip()]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Comma-separated list of allowed origins for CORS, e.g. "http://localhost:3000".
# Left permissive by default for local development against the Flutter app on
# an emulator/device; tighten this before any real deployment.
CORS_ORIGINS = _env_list("EID_CORS_ORIGINS", ["*"])

# Reject uploads larger than this (protects the server from oversized bodies).
MAX_UPLOAD_MB = int(os.environ.get("EID_MAX_UPLOAD_MB", "15"))

# Run OCR by default. Can be disabled globally (e.g. for a geometry-only
# deployment) without touching client code, since the flag is also
# accepted per-request.
DEFAULT_RUN_OCR = _env_bool("EID_DEFAULT_RUN_OCR", True)
DEFAULT_ENABLE_FORGERY_CHECK = _env_bool("EID_DEFAULT_ENABLE_FORGERY_CHECK", True)

# Whether to include the base64 rectified card image in responses (useful
# for the app to draw bbox overlays; adds payload size).
DEFAULT_INCLUDE_CANONICAL_IMAGE = _env_bool("EID_DEFAULT_INCLUDE_CANONICAL_IMAGE", True)

# Optional shared-secret API key. If set, requests must send it in the
# X-API-Key header. Unset (default) disables the check -- set this before
# exposing the API beyond localhost/dev.
API_KEY = os.environ.get("EID_API_KEY")  # None disables auth
