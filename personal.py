import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ── 1. Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhytoScan | Plant Disease Classifier",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── 2. Global CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Manrope:wght@300;400;500;600&display=swap');

/* ── Reset & Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d110e !important;
    color: #e8ede9 !important;
}
[data-testid="stAppViewContainer"] {
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(52,120,70,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 80% 80%, rgba(30,80,45,0.12) 0%, transparent 60%);
}
[data-testid="stHeader"], [data-testid="stToolbar"],
footer, #MainMenu { display: none !important; }

/* ── Typography ── */
* { font-family: 'Manrope', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }
code, .mono { font-family: 'DM Mono', monospace !important; }

/* ── Main container ── */
[data-testid="stMain"] > div:first-child { padding-top: 2rem; }
.block-container { max-width: 720px !important; padding: 0 1.5rem 4rem !important; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #5ecf7a;
    background: rgba(94,207,122,0.08);
    border: 1px solid rgba(94,207,122,0.22);
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    margin-bottom: 1.4rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: clamp(2.6rem, 6vw, 3.8rem) !important;
    font-weight: 400 !important;
    line-height: 1.1 !important;
    color: #f0f5f1 !important;
    margin: 0 0 0.6rem !important;
    letter-spacing: -0.02em;
}
.hero h1 em {
    font-style: italic;
    color: #5ecf7a;
}
.hero-sub {
    font-size: 0.95rem;
    color: #7a9982;
    font-weight: 300;
    letter-spacing: 0.02em;
    margin-top: 0.4rem;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(94,207,122,0.25), transparent);
    margin: 0.5rem 0 2.5rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1.5px dashed rgba(94,207,122,0.30) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: border-color 0.3s ease, background 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(94,207,122,0.55) !important;
    background: rgba(94,207,122,0.04) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span {
    color: #7a9982 !important;
    font-size: 0.88rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #5ecf7a !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(94,207,122,0.12) !important;
    border: 1px solid rgba(94,207,122,0.35) !important;
    color: #5ecf7a !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
}

/* ── Image display ── */
[data-testid="stImage"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(94,207,122,0.15);
    box-shadow: 0 8px 40px rgba(0,0,0,0.4);
}
[data-testid="stImage"] img {
    border-radius: 14px;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #5ecf7a !important;
}

/* ── Result cards ── */
.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1.8rem;
}
.result-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(94,207,122,0.15);
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #5ecf7a, #2a8a45);
}
.card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #5ecf7a;
    margin-bottom: 0.5rem;
}
.card-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #f0f5f1;
    line-height: 1.25;
}
.card-value.mono {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.7rem;
    color: #5ecf7a;
}
.crop-name {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #7a9982;
    margin-top: 0.3rem;
    letter-spacing: 0.04em;
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    vertical-align: middle;
}

/* ── Top predictions table ── */
.preds-section {
    margin-top: 1.4rem;
}
.preds-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #5ecf7a;
    margin-bottom: 0.9rem;
}
.pred-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.6rem;
}
.pred-rank {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #3d5e46;
    width: 1.4rem;
    text-align: right;
    flex-shrink: 0;
}
.pred-bar-wrap {
    flex: 1;
    height: 5px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
}
.pred-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #2a8a45, #5ecf7a);
}
.pred-label {
    font-size: 0.78rem;
    color: #a8c8b0;
    white-space: nowrap;
    max-width: 180px;
    overflow: hidden;
    text-overflow: ellipsis;
}
.pred-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #5ecf7a;
    width: 3.5rem;
    text-align: right;
    flex-shrink: 0;
}

/* ── Section labels ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #3d5e46;
    margin: 2.2rem 0 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ── 3. Load Model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('./plant_disease_cnn.keras')

model = load_model()

# ── 4. Class Names ───────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
]

def parse_class(raw: str):
    """Split 'Crop___Condition' into (crop, condition) with clean formatting."""
    parts = raw.split('___', 1)
    crop = parts[0].replace('_', ' ').replace('(', '(').strip()
    condition = parts[1].replace('_', ' ').strip() if len(parts) > 1 else 'Unknown'
    return crop, condition

def is_healthy(condition: str) -> bool:
    return 'healthy' in condition.lower()

# ── 5. Hero ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">CNN · TensorFlow · 38 Classes</div>
    <h1>Phyto<em>Scan</em></h1>
    <p class="hero-sub">Upload a leaf image — the model identifies crop &amp; disease in seconds.</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── 6. Upload ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Leaf Image Input</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop a leaf image here, or click to browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

# ── 7. Inference & Results ───────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.markdown('<p class="section-label">Target Image</p>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)

    # Preprocess
    img_rgb   = image.convert('RGB').resize((224, 224))
    img_array = np.array(img_rgb) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Inference
    with st.spinner("Running inference…"):
        predictions  = model.predict(img_batch, verbose=0)
        probs        = predictions[0]
        top_idx      = int(np.argmax(probs))
        confidence   = float(np.max(probs)) * 100
        crop, condition = parse_class(CLASS_NAMES[top_idx])
        healthy      = is_healthy(condition)

    dot_color  = "#5ecf7a" if healthy else "#e87c5a"
    status_txt = "Healthy" if healthy else "Disease Detected"

    # ── Result cards ──
    st.markdown(f"""
    <div class="result-grid">
        <div class="result-card">
            <div class="card-label">Classification</div>
            <div class="card-value">{condition}</div>
            <div class="crop-name">↳ {crop}</div>
        </div>
        <div class="result-card">
            <div class="card-label">Confidence Score</div>
            <div class="card-value mono">{confidence:.1f}<span style="font-size:1rem;color:#3d5e46">%</span></div>
            <div class="crop-name">
                <span class="status-dot" style="background:{dot_color}"></span>{status_txt}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top-5 predictions ──
    top5_idx  = np.argsort(probs)[::-1][:5]
    top5_vals = probs[top5_idx]
    bar_max   = float(top5_vals[0]) if top5_vals[0] > 0 else 1.0

    rows_html = ""
    for rank, (idx, val) in enumerate(zip(top5_idx, top5_vals), 1):
        pct      = val * 100
        bar_w    = (val / bar_max) * 100
        c_crop, c_cond = parse_class(CLASS_NAMES[idx])
        label    = f"{c_crop} · {c_cond}"
        opacity  = "1" if rank == 1 else "0.5"
        rows_html += (
            f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.75rem;opacity:{opacity};">' +
            f'<span style="font-family:DM Mono,monospace;font-size:0.65rem;color:#3d5e46;width:1.6rem;text-align:right;flex-shrink:0;">#{rank}</span>' +
            f'<div style="flex:1;height:5px;background:rgba(255,255,255,0.07);border-radius:100px;overflow:hidden;">' +
            f'<div style="width:{bar_w:.1f}%;height:100%;border-radius:100px;background:linear-gradient(90deg,#2a8a45,#5ecf7a);"></div></div>' +
            f'<span style="font-size:0.78rem;color:#a8c8b0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:200px;">{label}</span>' +
            f'<span style="font-family:DM Mono,monospace;font-size:0.72rem;color:#5ecf7a;width:3.5rem;text-align:right;flex-shrink:0;">{pct:.1f}%</span>' +
            '</div>'
        )

    header = '<div style="font-family:DM Mono,monospace;font-size:0.62rem;letter-spacing:0.16em;text-transform:uppercase;color:#5ecf7a;margin-bottom:1rem;">Top-5 Predictions</div>'
    st.markdown(f'<div style="margin-top:1.6rem;">{header}{rows_html}</div>', unsafe_allow_html=True)