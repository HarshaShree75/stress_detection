"""
app.py - Streamlit Frontend
============================
Run with:  streamlit run app.py

This is the main user interface for the Stress Detection System.
It allows users to upload a photo, processes it, and displays the result.
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from predict import predict_stress, load_model

# ─────────────────────────────────────────────
# PAGE CONFIGURATION  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Stress Detector",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  – clean, modern dark-accent theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Page background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    /* ── Main container ── */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 760px !important;
    }

    /* ── Hero title ── */
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e0c3fc, #8ec5fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        text-align: center;
        color: #8899aa;
        font-size: 1.0rem;
        font-weight: 300;
        margin-bottom: 2.2rem;
    }

    /* ── Upload card ── */
    .upload-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.6rem;
        backdrop-filter: blur(8px);
    }

    /* ── Result cards ── */
    .result-stressed {
        background: linear-gradient(135deg, rgba(255,75,75,0.15), rgba(200,50,50,0.05));
        border: 1px solid rgba(255,100,100,0.35);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        text-align: center;
    }
    .result-not-stressed {
        background: linear-gradient(135deg, rgba(50,205,100,0.15), rgba(30,140,70,0.05));
        border: 1px solid rgba(80,220,130,0.35);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        text-align: center;
    }
    .result-emoji { font-size: 3.8rem; margin-bottom: 0.4rem; }
    .result-label {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 0.2rem;
    }
    .result-confidence {
        font-size: 0.95rem;
        color: #aabbcc;
        font-weight: 300;
    }

    /* ── Info badge ── */
    .info-badge {
        display: inline-block;
        background: rgba(142,197,252,0.12);
        border: 1px solid rgba(142,197,252,0.25);
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        font-size: 0.78rem;
        color: #8ec5fc;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }

    /* ── Streamlit widgets overrides ── */
    .stFileUploader > div {
        background: rgba(255,255,255,0.03) !important;
        border: 1px dashed rgba(255,255,255,0.20) !important;
        border-radius: 12px !important;
        color: #aabbcc !important;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #8ec5fc, #e0c3fc) !important;
    }
    h3 { color: #dde8f8 !important; font-family: 'Syne', sans-serif !important; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] { background: #0d0d1a !important; }

    /* hide default Streamlit chrome */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 Stress Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Upload a facial photo — AI analyses your expression and detects stress in seconds.</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# SIDEBAR — How it works
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ How It Works")
    st.markdown("""
**Step 1 — Upload**
Upload any clear photo of a face (JPG, PNG, WEBP).

**Step 2 — Face Detection**
OpenCV locates the face in the image using a Haar Cascade detector.

**Step 3 — Preprocessing**
The face is converted to grayscale, resized to 48×48, and normalised.

**Step 4 — CNN Prediction**
A trained Convolutional Neural Network classifies the expression.

**Step 5 — Result**
The app shows *Stressed* or *Not Stressed* with a confidence score.

---
**Stress indicators (mapped from FER-2013):**
- 😡 Angry
- 😨 Fear  
- 🤢 Disgust

**Not-stressed indicators:**
- 😊 Happy
- 😐 Neutral
- 😲 Surprise
- 😢 Sad

---
*Model trained on FER-2013 dataset (35,887 images).*
    """)

# ─────────────────────────────────────────────
# MODEL STATUS CHECK
# ─────────────────────────────────────────────
model = load_model()

if model is None:
    st.warning(
        "⚠️ **Model not found.**  \n"
        "Please train the model first by running:\n"
        "```bash\n"
        "python model.py data/fer2013.csv\n"
        "```\n"
        "See the README for full setup instructions.",
        icon="⚠️",
    )

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
st.markdown("### 📁 Upload a Face Photo")

uploaded_file = st.file_uploader(
    label="Choose an image (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a clear, front-facing photo for best accuracy.",
    label_visibility="collapsed",
)

st.markdown(
    '<span class="info-badge">✓ JPG / PNG / WEBP</span>'
    '<span class="info-badge">✓ Front-facing preferred</span>'
    '<span class="info-badge">✓ Good lighting</span>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PROCESS & DISPLAY RESULTS
# ─────────────────────────────────────────────
if uploaded_file is not None:

    # ── Open uploaded image ──
    try:
        pil_image = Image.open(uploaded_file)
    except Exception:
        st.error("❌ Could not open the image. Please upload a valid JPG or PNG file.")
        st.stop()

    col1, col2 = st.columns([1, 1], gap="medium")

    # ── Left column: original image ──
    with col1:
        st.markdown("#### 🖼️ Uploaded Image")
        st.image(pil_image, use_container_width=True, caption="Original")

    # ── Analyse ──
    with st.spinner("🔍 Analysing facial expression…"):
        result = predict_stress(pil_image)

    # ── Right column: result ──
    with col2:
        st.markdown("#### 📊 Analysis Result")

        if not result["success"]:
            # ── Error ──
            st.error(f"**Error:** {result['error']}")

        else:
            label      = result["label"]
            confidence = result["confidence"]
            probs      = result["probabilities"]
            faces      = result["faces_found"]

            # ── Result card ──
            if label == "Stressed":
                emoji     = "😰"
                css_class = "result-stressed"
                tip       = "Take a deep breath — consider a short break, meditation, or a walk."
            else:
                emoji     = "😊"
                css_class = "result-not-stressed"
                tip       = "You appear calm and relaxed. Keep it up! 🌟"

            st.markdown(f"""
<div class="{css_class}">
  <div class="result-emoji">{emoji}</div>
  <div class="result-label">{label}</div>
  <div class="result-confidence">Confidence: <strong>{confidence:.1f}%</strong></div>
</div>
""", unsafe_allow_html=True)

            st.caption(f"💡 {tip}")
            st.caption(f"👤 Faces detected: {faces}")

    # ── Annotated image (full width) ──
    if result.get("success"):
        annotated_bgr = result["annotated_image"]
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        st.markdown("#### 🔲 Detected Face")
        st.image(annotated_rgb, use_container_width=True,
                 caption="Bounding box drawn around the analysed face")

        # ── Probability bar chart ──
        st.markdown("#### 📈 Confidence Breakdown")

        not_stressed_val = result["probabilities"]["Not Stressed"]
        stressed_val     = result["probabilities"]["Stressed"]

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Not Stressed", f"{not_stressed_val:.1f}%")
            st.progress(int(not_stressed_val))
        with col_b:
            st.metric("Stressed", f"{stressed_val:.1f}%")
            st.progress(int(stressed_val))

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#445566; font-size:0.8rem;'>"
    "Stress Detection System · Built with TensorFlow, OpenCV & Streamlit · "
    "Trained on FER-2013</p>",
    unsafe_allow_html=True,
)
