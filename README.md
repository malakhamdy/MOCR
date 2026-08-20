# Egyptian National ID — Arabic-First Extraction & Verification

## What this is

A scaffolded implementation of the pipeline described in the master prompt:
image normalization → card detection/rectification → front/back
classification → dynamic (scale-invariant) field localization →
field-specific preprocessing → Arabic-first OCR (PaddleOCR) → NID
structural validation → cross-field consistency → independent
verification → Streamlit UI with full visual debugging.

**This was built from scratch — no prior project existed to inspect
(Section 2's "first action" found nothing in the workspace).**

## Honest status (Section 49 — no fake success claims)

Built, run, and verified in this sandbox (no network access, no
PaddleOCR/Streamlit installed here):

| Component | Status |
|---|---|
| Image normalization (aspect-preserving, letterboxed, transform recorded) | ✅ implemented + tested |
| Card detection (contour/quad) + perspective rectification | ✅ implemented + tested |
| Canonical-card coordinate system, scale-invariant localization | ✅ implemented + tested (synthetic fixtures, see caveat below) |
| Front/back classification | ✅ implemented (heuristic; needs real-card calibration) |
| Template registry with normalized field regions | ✅ implemented, **regions are a coarse starting prior, not calibrated against real cards** |
| Field-specific preprocessing variants | ✅ implemented |
| Arabic normalization (digits, tatweel, whitespace) | ✅ implemented + tested |
| Egyptian NID structural validator (length/date/governorate) | ✅ implemented + tested |
| NID checksum | ⚠️ implemented but **unverified against an official spec** — flagged explicitly in code and results, per Section 1 ("do not hallucinate") |
| Cross-field validation (DOB/gender/governorate vs NID-derived) | ✅ implemented |
| Independent verification layer (EXTRACTED vs VALIDATED vs VERIFIED) | ✅ implemented |
| Barcode/PDF417 detection + decode | ✅ implemented (pyzbar-based, degrades gracefully if pyzbar absent) |
| PaddleOCR integration | ✅ written against the API, version-defensive init — **not runtime-tested here**, this sandbox has no network/PaddleOCR install |
| Streamlit app | ✅ written, syntax-checked — **not launched here** (no `streamlit` package, no network) |
| Automated tests | ✅ 27 tests, all passing in this sandbox (NID validator, Arabic normalization, image normalization, scale-invariance) |

## What is NOT yet true, and must not be assumed true

1. **The template field regions (`core/templates.py`) are a best-effort
   geometric prior**, not calibrated against real, physical Egyptian ID
   cards. You must run the pipeline against real (or realistic) sample
   images and adjust the `NormalizedBBox` fractions until crops line up.
2. **OCR has never actually run.** `core/ocr_engine.py` is written
   defensively against the PaddleOCR API (with fallback init attempts
   across versions) but has not executed a single real inference in this
   environment. Install `paddleocr`/`paddlepaddle` and validate against
   real fixtures before trusting it.
3. **The NID checksum algorithm is unverified** — see the prominent
   caveat in `core/nid_validator.py`. Structural checks (length, date,
   governorate) are based on well-established public knowledge of the
   format; the checksum is a commonly-circulated but not officially
   confirmed algorithm, and failures are reported as a distinct,
   lower-trust status rather than folded into a blanket "invalid".
4. **The scale-invariance tests use a synthetic fixture**
   (`tests/fixtures/synthetic_card.py`) — a plain rectangle with crude
   interior marks, not a real ID photo — because none was available in
   this sandbox. They prove the *geometry pipeline* is scale-invariant;
   they do not prove real-world OCR accuracy across resolutions
   (Section 53's full acceptance test still needs real card photos).
5. **Front/back classification is a coarse heuristic** (photo-block color
   variance vs. barcode-block edge density) and needs real samples to
   validate/tune.

## Running it for real

```bash
pip install -r requirements.txt
pytest tests/                       # includes tests that need pytest
streamlit run app.py
```

(In this sandbox, `pytest` itself wasn't installable offline, so
`tests/run_tests_no_pytest.py` was used as a zero-dependency runner to
actually execute the test logic here. The test *files* are normal pytest
tests — just run them with real `pytest` once you have network access.)

## Next steps (in priority order)

1. Get real (or high-quality synthetic/redacted) front & back ID sample
   images and recalibrate `core/templates.py` regions against them.
2. Install PaddleOCR and validate `core/ocr_engine.py` actually returns
   sensible Arabic text on real crops; adjust the version-compatibility
   fallback list in `get_engine()` based on what's actually installed.
3. Run the Section 53 acceptance test (multiple versions of the same
   physical card: low-res, rotated, perspective-distorted, etc.) with a
   real card, not the synthetic fixture.
4. If a second local OCR engine or barcode library is available, wire it
   in as `independent_verifier`'s two-pass agreement source.
5. Try to independently confirm (or replace) the NID checksum algorithm
   against an authoritative source.
