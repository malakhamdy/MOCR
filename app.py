"""
Streamlit UI for Egyptian National ID Extraction & Verification.

Supports Arabic-first OCR extraction, card detection, homography rectification,
NID checksum & mathematical validation, cross-field consistency, and visual debugging.
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import json

from core import config
from core.pipeline import run_pipeline
from core import ocr_engine

st.set_page_config(
    page_title="Egyptian National ID — OCR & Verification",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🪪 Egyptian National ID — Extraction & Verification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Arabic-First OCR, Geometric Rectification, Structural Rule Checks & Cross-Field Consistency</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("⚙️ Configuration & Controls")

side_override_opt = st.sidebar.selectbox(
    "Card Side Selection",
    options=["Auto-detect", "Front (الوجه الأمامي)", "Back (الوجه الخلفي)"],
    index=0,
    help="Select 'Auto-detect' to automatically classify the card side based on visual features, or force a specific side."
)
force_side = None
if "Front" in side_override_opt:
    force_side = config.SIDE_FRONT
elif "Back" in side_override_opt:
    force_side = config.SIDE_BACK

enable_ocr = st.sidebar.toggle("Run OCR Engine", value=True)

loc_mode_opt = st.sidebar.radio(
    "Field Localization Engine",
    options=["Dynamic Anchors (Default)", "YOLO Segmentation (AI Model)"],
    index=0,
    help="Dynamic Anchors uses semantic visual landmarks and ink projection. YOLO Segmentation uses a trained deep learning segmentation network."
)
localization_mode = config.LOCALIZATION_MODE_YOLO if "YOLO" in loc_mode_opt else config.LOCALIZATION_MODE_ANCHORS

if not ocr_engine.is_available():
    st.sidebar.warning(
        f"⚠️ PaddleOCR is not installed in this environment. Only detection/geometry will run. "
        f"Error: {ocr_engine.import_error()}"
    )
else:
    st.sidebar.success("✅ PaddleOCR Engine Ready")

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Demo & Testing")
use_demo = st.sidebar.button("Load Sample Demo Card")

uploaded = st.file_uploader("Upload photo or scan of Egyptian ID (Front or Back)", type=["jpg", "jpeg", "png"])

image_bgr = None

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
elif use_demo:
    # Generate a clean demo card
    sample = np.full((630, 1000, 3), 245, dtype=np.uint8)
    cv2.rectangle(sample, (30, 30), (970, 600), (220, 225, 230), -1)
    cv2.rectangle(sample, (50, 150), (280, 560), (160, 180, 200), -1)
    cv2.putText(sample, "PHOTO AREA", (80, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
    cv2.putText(sample, "Ahmed Mohamed Ali", (320, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    cv2.putText(sample, "29501011234567", (320, 540), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    image_bgr = sample

if image_bgr is not None:
    with st.spinner("Processing Egyptian ID through pipeline..."):
        result = run_pipeline(
            image_bgr,
            run_ocr=enable_ocr,
            force_side=force_side,
            localization_mode=localization_mode
        )

    doc = result.document

    # --- 1. Detection Summary ------------------------------------------------
    st.subheader("1. Card Detection & Classification")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Card Detected", "✅ Yes" if doc.is_egyptian_id else "❌ No")
    m2.metric("Detection Confidence", f"{doc.card_detection_confidence:.1%}")
    m3.metric("Card Side", f"{doc.side.upper()} ({doc.side_confidence:.1%})" if doc.side else "Unknown")
    m4.metric("Localization Engine", result.debug.get("localization_mode", "dynamic_anchors").replace("_", " ").title())

    if result.errors:
        for e in result.errors:
            st.error(f"⚠️ {e}")

    # --- 2. Extraction Results -----------------------------------------------
    st.subheader("2. Extracted Fields & Verification")
    if result.fields:
        rows = []
        for name, fr in result.fields.items():
            ocr_conf = f"{fr.ocr_confidence:.1%}" if fr.ocr_confidence is not None else "-"
            loc_conf = f"{fr.localization_confidence:.1%}" if fr.localization_confidence is not None else "-"
            issues_str = ", ".join(fr.issues) if fr.issues else "None"
            rows.append({
                "Field Name": name,
                "Extracted Value": fr.normalized or fr.raw or "-",
                "Raw OCR": fr.raw or "-",
                "Status": fr.status,
                "OCR Conf.": ocr_conf,
                "Localization Conf.": loc_conf,
                "Detection Method": getattr(fr, "detection_method", "dynamic_anchor"),
                "Issues": issues_str,
            })
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No fields extracted from this card image.")

    # --- 3. Derived Insights -------------------------------------------------
    if result.derived:
        st.subheader("3. Decoded Identity Data (from National ID Number)")
        d_cols = st.columns(max(len(result.derived), 1))
        col_idx = 0
        for key, val in result.derived.items():
            with d_cols[col_idx % len(d_cols)]:
                label = key.replace("_", " ").title()
                if isinstance(val, dict):
                    st.write(f"**{label}**")
                    st.json(val)
                else:
                    st.metric(label, str(val))
            col_idx += 1

    # --- 4. Cross-Field Validation -------------------------------------------
    if result.cross_field_validation:
        st.subheader("4. Cross-Field Consistency Checks")
        st.json(result.cross_field_validation)

    # --- 5. Visual Debugging Sequence ----------------------------------------
    st.subheader("5. Visual Debugging Pipeline")
    tabs = st.tabs([
        "📸 Original Upload",
        "📐 Canonical Rectified Card",
        "🎯 Localized Field Bounding Boxes / Polygons",
        "🔍 Field Crops Preview",
        "⚙️ Normalization Metadata",
    ])

    with tabs[0]:
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Original Upload Image", use_container_width=True)

    canonical_card = result.canonical_card
    if canonical_card is None and doc.is_egyptian_id:
        from core.image_utils import normalize_image
        from core.card_detection import detect_and_rectify, crop_canonical_to_card
        proc_img, _, _ = normalize_image(image_bgr)
        det = detect_and_rectify(proc_img)
        if det["canonical_card"] is not None:
            canonical_card = crop_canonical_to_card(det["canonical_card"])

    with tabs[1]:
        if canonical_card is not None:
            st.image(cv2.cvtColor(canonical_card, cv2.COLOR_BGR2RGB), caption="Canonical Card (1600 x 1009 ID-1 Rectified)", use_container_width=True)
        else:
            st.info("Canonical card could not be rectified.")

    with tabs[2]:
        if canonical_card is not None and result.fields:
            annotated = canonical_card.copy()
            overlay = annotated.copy()
            palette = [
                (0, 200, 0), (200, 100, 0), (0, 100, 200), (180, 0, 180),
                (0, 180, 180), (220, 140, 20), (50, 200, 100), (120, 80, 220)
            ]
            for idx, (name, fr) in enumerate(result.fields.items()):
                color = palette[idx % len(palette)]
                # Draw polygon if available (YOLO Segmentation)
                if fr.polygon and len(fr.polygon) >= 3:
                    pts = np.array(fr.polygon, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

                # Draw bounding box
                if fr.bbox:
                    bbox_coords = fr.bbox.get("bbox") if isinstance(fr.bbox, dict) else fr.bbox
                    if bbox_coords and len(bbox_coords) == 4:
                        x, y, w, h = [int(v) for v in bbox_coords]
                        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(annotated, f"{name}", (x, max(y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Localized Fields & Segmentation Masks", use_container_width=True)
        else:
            st.info("No field bounding boxes available.")

    with tabs[3]:
        if result.fields:
            crop_cols = st.columns(min(len(result.fields), 4))
            for i, (fname, fr) in enumerate(result.fields.items()):
                with crop_cols[i % len(crop_cols)]:
                    if canonical_card is not None and fr.bbox:
                        bbox_coords = fr.bbox.get("bbox") if isinstance(fr.bbox, dict) else fr.bbox
                        if bbox_coords and len(bbox_coords) == 4:
                            x, y, w, h = [int(v) for v in bbox_coords]
                            crop = canonical_card[y:y+h, x:x+w]
                            if crop.size > 0:
                                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=f"Crop: {fname}")
                            else:
                                st.caption(f"{fname}: empty crop")

    with tabs[4]:
        st.json(result.debug)

    # --- 6. Export Results ---------------------------------------------------
    st.subheader("6. Export Structured Output")
    st.download_button(
        label="📥 Download Structured Result (JSON)",
        data=json.dumps(result.as_dict(), indent=2, ensure_ascii=False),
        file_name="egyptian_id_ocr_result.json",
        mime="application/json",
    )

else:
    st.info("👆 Upload a photo of an Egyptian National ID (Front or Back) or click 'Load Sample Demo Card' in the sidebar to test.")
