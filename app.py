"""
Streamlit UI (Sections 38-40).

NOTE: this app requires `streamlit`, `opencv-python`, and (for OCR)
`paddleocr`/`paddlepaddle` to be installed in the environment it's run in.
It was authored and structurally reviewed here but could not be launched
in this sandbox (no network access to install those packages) -- run it
locally with `streamlit run app.py` after `pip install -r requirements.txt`.
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image

from core import config
from core.pipeline import run_pipeline
from core import ocr_engine

st.set_page_config(page_title="Egyptian National ID Extraction", layout="wide")
st.title("Egyptian National ID — Arabic-First Extraction & Verification")

if not ocr_engine.is_available():
    st.warning(
        "PaddleOCR is not installed in this environment, so OCR steps will be "
        "skipped and only detection/localization/geometry results shown. "
        f"Import error: {ocr_engine.import_error()}"
    )

uploaded = st.file_uploader("Upload a photo of the ID (front or back)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    with st.spinner("Running pipeline..."):
        result = run_pipeline(image_bgr, run_ocr=True)

    doc = result.document

    # --- Detection summary -------------------------------------------------
    st.header("1. Detection")
    c1, c2, c3 = st.columns(3)
    c1.metric("Card detected", "Yes" if doc.is_egyptian_id else "No")
    c2.metric("Card confidence", f"{doc.card_detection_confidence:.2f}")
    c3.metric("Side", f"{doc.side} ({doc.side_confidence:.2f})")

    if result.errors:
        for e in result.errors:
            st.error(e)

    # --- Visual debugging pipeline (Section 39) -----------------------------
    st.header("2. Visual Debug Sequence")
    tabs = st.tabs(["Original", "Processing canvas", "Card corners", "Canonical card", "Field boxes"])

    with tabs[0]:
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Original upload")

    with tabs[1]:
        norm_meta = result.debug.get("normalization_transform", {})
        st.json(norm_meta)

    with tabs[2]:
        corners = result.debug.get("card_detection", {}).get("corners")
        st.write(f"Detected corners (processing-canvas coords): {corners}")

    canonical_card = None
    if doc.is_egyptian_id:
        # Recompute for display purposes only (pipeline doesn't return the
        # image object in the lightweight debug dict to keep it JSON-clean)
        from core.image_utils import normalize_image
        from core.card_detection import detect_and_rectify
        proc_img, _, _ = normalize_image(image_bgr)
        det = detect_and_rectify(proc_img)
        canonical_card = det["canonical_card"]

    with tabs[3]:
        if canonical_card is not None:
            st.image(cv2.cvtColor(canonical_card, cv2.COLOR_BGR2RGB), caption="Canonical rectified card")
        else:
            st.info("No canonical card available (detection failed).")

    with tabs[4]:
        if canonical_card is not None and result.fields:
            annotated = canonical_card.copy()
            for name, fr in result.fields.items():
                if fr.bbox:
                    x, y, w, h = fr.bbox["bbox"]
                    cv2.rectangle(annotated, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
                    cv2.putText(annotated, name, (int(x), max(int(y) - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Localized field boxes")
        else:
            st.info("No fields localized.")

    # --- Extraction table (Section 38) --------------------------------------
    st.header("3. Extraction Results")
    if result.fields:
        rows = []
        for name, fr in result.fields.items():
            rows.append({
                "Field": name,
                "Value (normalized)": fr.normalized,
                "OCR Confidence": f"{fr.ocr_confidence:.2f}" if fr.ocr_confidence is not None else "-",
                "Localization Conf.": f"{fr.localization_confidence:.2f}" if fr.localization_confidence is not None else "-",
                "Status": fr.status,
                "Issues": ", ".join(fr.issues) if fr.issues else "-",
            })
        st.table(rows)
    else:
        st.info("No fields extracted.")

    # --- Verification narrative (Section 38) --------------------------------
    st.header("4. Verification Detail")
    for name, fr in result.fields.items():
        with st.expander(f"{name} — status: {fr.status}"):
            st.write("**Raw OCR:**", fr.raw)
            st.write("**Normalized:**", fr.normalized)
            if fr.validation:
                st.write("**Validation:**")
                st.json(fr.validation)
            if fr.verification:
                st.write("**Verification:**")
                st.json(fr.verification)

    if result.cross_field_validation:
        st.header("5. Cross-Field Consistency")
        st.json(result.cross_field_validation)

    # --- Debug expander (Section 38) -----------------------------------------
    st.header("6. Debug")
    with st.expander("Full debug payload (preprocessing, OCR candidates, transforms)"):
        st.json(result.debug)

    with st.expander("Full structured result (JSON)"):
        st.json(result.as_dict())
else:
    st.info("Upload a front or back image of an Egyptian National ID to begin.")
