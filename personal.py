import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import json

# ── 1. Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhytoScan | Plant Disease Classifier",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── 2. Global CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Manrope:wght@300;400;500;600;700&display=swap');

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

/* ── Hiding Elements Safely ── */
/* Keep the stToolbar alive so the toggle button renders, but hide its unwanted siblings */

[data-testid="stAppDeployButton"], /* Hides the "Deploy" text/button */
[data-testid="stMainMenu"],        /* Hides the 3-dot menu */
[data-testid="stToolbarActions"],  /* Hides the empty action container */
footer { 
    display: none !important; 
}

/* Make the header and toolbar containers completely transparent */
[data-testid="stHeader"], 
[data-testid="stToolbar"] {
    background-color: transparent !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0a0e0b !important;
    border-right: 1px solid rgba(94,207,122,0.10) !important;
}
[data-testid="stSidebar"] * {
    color: #c0d4c5 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 0.8rem !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(94,207,122,0.08) !important;
}
[data-testid="stSidebar"] .stRadio [data-checked="true"] label {
    background: rgba(94,207,122,0.12) !important;
    color: #5ecf7a !important;
    font-weight: 600 !important;
}
            
/* ── Force Sidebar Toggle Visibility ── */
[data-testid="collapsedControl"], 
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    z-index: 999999 !important;
    background-color: transparent !important;
    transition: background-color 0.3s ease !important;
}

/* Color the SVG icon so it doesn't blend into the dark background */
[data-testid="collapsedControl"] svg, 
[data-testid="stSidebarCollapsedControl"] svg, 
[data-testid="stHeader"] svg {
    color: #5ecf7a !important;
    fill: #5ecf7a !important;
}

/* Add a slight hover effect to the toggle */
[data-testid="collapsedControl"]:hover, 
[data-testid="stSidebarCollapsedControl"]:hover {
    background-color: rgba(94, 207, 122, 0.1) !important;
    border-radius: 8px;
}

/* ── Typography ── */
* { font-family: 'Manrope', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }
code, .mono { font-family: 'DM Mono', monospace !important; }

/* ── Main container ── */
[data-testid="stMain"] > div:first-child { padding-top: 1rem; }
.block-container { max-width: 780px !important; padding: 0 1.5rem 4rem !important; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #5ecf7a;
    background: rgba(94,207,122,0.08);
    border: 1px solid rgba(94,207,122,0.22);
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: clamp(2.4rem, 5.5vw, 3.4rem) !important;
    font-weight: 400 !important;
    line-height: 1.1 !important;
    color: #f0f5f1 !important;
    margin: 0 0 0.5rem !important;
    letter-spacing: -0.02em;
}
.hero h1 em {
    font-style: italic;
    color: #5ecf7a;
}
.hero-sub {
    font-size: 0.92rem;
    color: #7a9982;
    font-weight: 300;
    letter-spacing: 0.02em;
    margin-top: 0.3rem;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(94,207,122,0.25), transparent);
    margin: 0.5rem 0 2rem;
}

/* ── Section headings ── */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #f0f5f1;
    margin: 0 0 0.3rem;
    line-height: 1.2;
}
.section-subtitle {
    font-size: 0.85rem;
    color: #5a7e63;
    font-weight: 400;
    margin-bottom: 1.5rem;
}
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #3d5e46;
    margin: 2.2rem 0 0.9rem;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(94,207,122,0.12);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #5ecf7a, #2a8a45);
}
.glass-card-muted {
    background: rgba(255,255,255,0.015);
    border: 1px solid rgba(94,207,122,0.08);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 0.8rem;
}

/* ── Stat pill ── */
.stat-row {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.stat-pill {
    background: rgba(94,207,122,0.06);
    border: 1px solid rgba(94,207,122,0.15);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    flex: 1;
    min-width: 120px;
    text-align: center;
}
.stat-num {
    font-family: 'DM Mono', monospace;
    font-size: 1.5rem;
    color: #5ecf7a;
    font-weight: 500;
    line-height: 1.2;
}
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5a7e63;
    margin-top: 0.3rem;
}

/* ── Result cards (classifier page) ── */
.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1.5rem;
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

/* ── Layer table ── */
.layer-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin: 1rem 0;
}
.layer-table th {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5ecf7a;
    padding: 0.6rem 0.8rem;
    text-align: left;
    border-bottom: 1px solid rgba(94,207,122,0.15);
    background: rgba(94,207,122,0.04);
}
.layer-table th:first-child { border-radius: 8px 0 0 0; }
.layer-table th:last-child { border-radius: 0 8px 0 0; }
.layer-table td {
    font-size: 0.8rem;
    color: #c0d4c5;
    padding: 0.55rem 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.layer-table tr:last-child td { border-bottom: none; }
.layer-table td:first-child { font-family: 'DM Mono', monospace; color: #a8c8b0; }
.layer-table td:nth-child(2) { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #7a9982; }
.layer-table td:nth-child(3) { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #5ecf7a; }

/* ── Epoch bar chart ── */
.epoch-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.35rem;
}
.epoch-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    color: #5a7e63;
    width: 1.6rem;
    text-align: right;
    flex-shrink: 0;
}
.epoch-bar-bg {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.04);
    border-radius: 100px;
    overflow: hidden;
}
.epoch-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.6s ease;
}
.epoch-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #a8c8b0;
    width: 3rem;
    text-align: right;
    flex-shrink: 0;
}

/* ── Disease info card ── */
.disease-info {
    background: rgba(232,124,90,0.06);
    border: 1px solid rgba(232,124,90,0.18);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-top: 1rem;
}
.disease-info-healthy {
    background: rgba(94,207,122,0.06);
    border: 1px solid rgba(94,207,122,0.18);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-top: 1rem;
}
.info-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #e87c5a;
    margin-bottom: 0.5rem;
}
.info-title-healthy {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5ecf7a;
    margin-bottom: 0.5rem;
}
.info-text {
    font-size: 0.82rem;
    color: #c0d4c5;
    line-height: 1.55;
}

/* ── Timeline ── */
.timeline-item {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.2rem;
    position: relative;
}
.timeline-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #5ecf7a;
    margin-top: 0.35rem;
    flex-shrink: 0;
    box-shadow: 0 0 8px rgba(94,207,122,0.3);
}
.timeline-content {
    flex: 1;
    padding-bottom: 1rem;
    border-left: 1px solid rgba(94,207,122,0.12);
    padding-left: 1rem;
    margin-left: -0.55rem;
}
.timeline-step {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5ecf7a;
    margin-bottom: 0.2rem;
}
.timeline-desc {
    font-size: 0.84rem;
    color: #a8c8b0;
    line-height: 1.5;
}

/* ── Team card ── */
.team-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin: 1rem 0;
}
.team-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(94,207,122,0.10);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    transition: border-color 0.3s ease, transform 0.2s ease;
}
.team-card:hover {
    border-color: rgba(94,207,122,0.3);
    transform: translateY(-2px);
}
.team-emoji {
    font-size: 1.8rem;
    margin-bottom: 0.4rem;
}
.team-name {
    font-family: 'DM Serif Display', serif;
    font-size: 0.95rem;
    color: #f0f5f1;
    margin-bottom: 0.15rem;
}
.team-role {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #5a7e63;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    margin-top: 3rem;
    padding: 1.5rem 0;
    border-top: 1px solid rgba(94,207,122,0.08);
}
.footer-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    color: #3d5e46;
}
</style>
""", unsafe_allow_html=True)


# ── 3. Load Model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('./plant_disease_cnn.keras')

model = load_model()

# ── 4. Class Names & Helpers ─────────────────────────────────────────────────
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

# Build crop→diseases lookup
CROP_DISEASES = {}
for c in CLASS_NAMES:
    parts = c.split('___', 1)
    crop = parts[0].replace('_', ' ').replace('(', '(').strip()
    cond = parts[1].replace('_', ' ').strip() if len(parts) > 1 else 'Unknown'
    CROP_DISEASES.setdefault(crop, []).append(cond)

UNIQUE_CROPS = sorted(CROP_DISEASES.keys())
DISEASE_COUNT = sum(1 for c in CLASS_NAMES if 'healthy' not in c.lower())
HEALTHY_COUNT = sum(1 for c in CLASS_NAMES if 'healthy' in c.lower())

# Disease brief descriptions for the classifier page
DISEASE_INFO = {
    'Apple scab': 'A fungal disease caused by Venturia inaequalis. It creates dark, scabby lesions on leaves and fruit. Treat with fungicide sprays and remove fallen infected leaves.',
    'Black rot': 'Caused by the fungus Botryosphaeria obtusa. It produces brown circular lesions on leaves and mummified fruit. Prune dead wood and apply fungicides during wet weather.',
    'Cedar apple rust': 'A fungal disease requiring two hosts (cedar and apple). Orange-yellow spots appear on apple leaves. Remove nearby cedar trees or apply preventive fungicide.',
    'Powdery mildew': 'A white powdery fungal coating on leaves and stems. Caused by various fungi. Improve air circulation and apply sulfur-based or potassium bicarbonate sprays.',
    'Cercospora leaf spot Gray leaf spot': 'A fungal disease of corn that creates rectangular gray lesions. Rotate crops, use resistant hybrids, and apply foliar fungicides.',
    'Common rust': 'Caused by Puccinia sorghi. Small, circular, reddish-brown pustules form on both leaf surfaces. Use resistant varieties and apply fungicides if severe.',
    'Northern Leaf Blight': 'Caused by Exserohilum turcicum. Long, cigar-shaped gray-green lesions appear on corn leaves. Use resistant hybrids and practice crop rotation.',
    'Esca (Black Measles)': 'A complex fungal disease of grapevines. Causes dark spots on berries and tiger-stripe patterns on leaves. No cure exists; remove infected vines.',
    'Leaf blight (Isariopsis Leaf Spot)': 'A fungal disease causing angular brown lesions on grape leaves. Apply preventive fungicides and maintain good canopy management.',
    'Haunglongbing (Citrus greening)': 'A devastating bacterial disease spread by psyllid insects. Causes misshapen, bitter fruit and yellow mottled leaves. No cure; control the insect vector.',
    'Bacterial spot': 'Caused by Xanthomonas bacteria. Creates dark, water-soaked spots on leaves and fruit. Use copper-based sprays and disease-free seeds.',
    'Early blight': 'Caused by Alternaria solani. Creates concentric ring patterns (target spots) on lower leaves. Rotate crops, mulch, and apply fungicides.',
    'Late blight': 'Caused by Phytophthora infestans — the same pathogen behind the Irish Potato Famine. Dark, water-soaked lesions spread rapidly. Apply fungicides immediately.',
    'Leaf Mold': 'A fungal disease caused by Passalora fulva. Yellow patches on upper leaf surface with olive-green mold beneath. Improve ventilation in greenhouses.',
    'Septoria leaf spot': 'Caused by Septoria lycopersici. Small circular spots with dark borders and gray centers on lower leaves. Remove infected leaves and apply fungicide.',
    'Spider mites Two-spotted spider mite': 'Not a disease but a pest. Tiny mites cause stippled, yellowed leaves with fine webbing. Use miticides or introduce predatory mites.',
    'Target Spot': 'Caused by Corynespora cassiicola. Creates brown concentric-ringed spots on tomato leaves. Use resistant varieties and apply fungicides.',
    'Tomato mosaic virus': 'A viral disease causing mottled yellow-green leaves and stunted growth. No chemical treatment; use resistant varieties and sanitize tools.',
    'Tomato Yellow Leaf Curl Virus': 'Spread by whiteflies. Causes upward curling of leaves, yellowing, and stunted plants. Control whitefly populations and use resistant varieties.',
    'Leaf scorch': 'Caused by the fungus Diplocarpon earlianum. Irregular purple-red spots that merge and cause leaf edges to dry up. Remove infected leaves and apply fungicides.',
}

def parse_class(raw: str):
    """Split 'Crop___Condition' into (crop, condition) with clean formatting."""
    parts = raw.split('___', 1)
    crop = parts[0].replace('_', ' ').replace('(', '(').strip()
    condition = parts[1].replace('_', ' ').strip() if len(parts) > 1 else 'Unknown'
    return crop, condition

def is_healthy(condition: str) -> bool:
    return 'healthy' in condition.lower()

# Training history from the notebook
TRAINING_HISTORY = {
    'accuracy':     [0.4206, 0.6045, 0.6778, 0.7257, 0.7593, 0.7862, 0.8145, 0.8415, 0.8530, 0.8650, 0.8784, 0.8830, 0.8936, 0.8985, 0.9055, 0.9110, 0.9177],
    'val_accuracy': [0.7031, 0.8171, 0.8500, 0.8292, 0.8831, 0.8822, 0.8937, 0.9065, 0.9122, 0.9168, 0.9190, 0.9239, 0.9245, 0.9246, 0.9023, 0.9233, 0.9246],
    'loss':         [2.1004, 1.2980, 1.0144, 0.8508, 0.7315, 0.6453, 0.5563, 0.4732, 0.4313, 0.3910, 0.3560, 0.3437, 0.3054, 0.2953, 0.2722, 0.2600, 0.2365],
    'val_loss':     [1.0245, 0.6540, 0.5069, 0.5440, 0.3744, 0.3744, 0.3376, 0.3123, 0.2996, 0.2942, 0.2692, 0.2640, 0.2767, 0.2934, 0.3667, 0.2936, 0.2869],
}


# ── 5. Sidebar Navigation ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.2rem 0 1.5rem;">
        <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#f0f5f1;">
            🌿 Phyto<em style="color:#5ecf7a;">Scan</em>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:0.55rem;letter-spacing:0.15em;color:#3d5e46;margin-top:0.3rem;">
            GROUP 6 · CAPSTONE PROJECT
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Overview", "📊  Dataset", "🧠  Model Architecture", "📈  Training Results", "🔬  Live Classifier", "👥  Team"],
        label_visibility="collapsed",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">CNN · TensorFlow · 38 Classes</div>
        <h1>Phyto<em>Scan</em></h1>
        <p class="hero-sub">AI-powered plant disease identification from a single leaf photo</p>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    # Problem statement
    st.markdown('<div class="section-title">The Problem</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Why this project matters</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="info-text">
            Smallholder farmers produce up to <strong style="color:#5ecf7a;">80%</strong> of food in developing regions,
            yet lose an estimated <strong style="color:#e87c5a;">20–40%</strong> of their crop yields to pests and pathogens every year.
            <br><br>
            Traditionally, diagnosing plant diseases requires a university-trained expert to visit the field in person.
            These experts are critically scarce in developing countries — diseases go unnoticed until an entire harvest is ruined.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Our solution
    st.markdown('<div class="section-title" style="margin-top:2rem;">Our Solution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Democratizing agricultural expertise</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="info-text">
            We built a Convolutional Neural Network from scratch that identifies <strong style="color:#5ecf7a;">38 different plant conditions</strong>
            across <strong style="color:#5ecf7a;">14 crop species</strong> from a single leaf photograph.
            <br><br>
            The goal is to put an automated, instant, and free diagnostic tool directly into farmers' hands via basic smartphones.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key stats
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-pill">
            <div class="stat-num">54,305</div>
            <div class="stat-label">Leaf Images</div>
        </div>
        <div class="stat-pill">
            <div class="stat-num">38</div>
            <div class="stat-label">Classes</div>
        </div>
        <div class="stat-pill">
            <div class="stat-num">14</div>
            <div class="stat-label">Crop Species</div>
        </div>
        <div class="stat-pill">
            <div class="stat-num">92.5%</div>
            <div class="stat-label">Val Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline timeline
    st.markdown('<div class="section-title" style="margin-top:2.5rem;">Project Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">End-to-end machine learning workflow</div>', unsafe_allow_html=True)

    steps = [
        ("Step 1 · Data Acquisition", "Pulled the PlantVillage dataset directly from GitHub using sparse checkout to grab only the raw/color folder we needed."),
        ("Step 2 · Preprocessing", "Resized images to 224×224, normalized pixel values from 0–255 to 0–1, and applied parallel processing with caching and prefetching."),
        ("Step 3 · Model Building", "Built a 3-block CNN from scratch with Conv2D, MaxPooling, Flatten, Dense layers, and 50% Dropout to fight overfitting."),
        ("Step 4 · Training", "Trained for 17 epochs with EarlyStopping (patience=5) and ModelCheckpoint. Best weights restored from Epoch 12."),
        ("Step 5 · Evaluation", "Plotted accuracy/loss curves and validated model performance at 92.46% validation accuracy."),
    ]
    timeline_html = ""
    for step_label, desc in steps:
        timeline_html += f"""
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <div class="timeline-step">{step_label}</div>
                <div class="timeline-desc">{desc}</div>
            </div>
        </div>"""
    st.markdown(timeline_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Dataset
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Dataset":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">PlantVillage · 54,305 Images</div>
        <h1>Dataset <em>Explorer</em></h1>
        <p class="hero-sub">Browse the 14 crops and 38 classes used for training</p>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    # Dataset source
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">Data Source</div>
        <div class="info-text">
            The <strong style="color:#5ecf7a;">PlantVillage</strong> dataset was created by researchers
            <strong>David Hughes</strong> and <strong>Marcel Salathé</strong> at Penn State University and EPFL (Switzerland) around 2015.
            <br><br>
            It provides a standardized, open-access collection of leaf images taken under controlled conditions
            to serve as a benchmark for training plant disease classifiers.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-pill">
            <div class="stat-num">{len(UNIQUE_CROPS)}</div>
            <div class="stat-label">Crops</div>
        </div>
        <div class="stat-pill">
            <div class="stat-num">{DISEASE_COUNT}</div>
            <div class="stat-label">Diseases</div>
        </div>
        <div class="stat-pill">
            <div class="stat-num">{HEALTHY_COUNT}</div>
            <div class="stat-label">Healthy Classes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Crop breakdown
    st.markdown('<div class="section-title" style="margin-top:2rem;">Classes per Crop</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Select a crop to see its disease classes</div>', unsafe_allow_html=True)

    selected_crop = st.selectbox("Select a crop", UNIQUE_CROPS, label_visibility="collapsed")

    if selected_crop:
        conditions = CROP_DISEASES[selected_crop]
        cards_html = ""
        for cond in conditions:
            if 'healthy' in cond.lower():
                dot = '<span class="status-dot" style="background:#5ecf7a;"></span>'
                badge_color = "rgba(94,207,122,0.12)"
                border_color = "rgba(94,207,122,0.20)"
                text_color = "#5ecf7a"
            else:
                dot = '<span class="status-dot" style="background:#e87c5a;"></span>'
                badge_color = "rgba(232,124,90,0.08)"
                border_color = "rgba(232,124,90,0.18)"
                text_color = "#e87c5a"

            cards_html += f"""
<div style="background:{badge_color};border:1px solid {border_color};border-radius:10px;padding:0.7rem 1rem;margin-bottom:0.5rem;display:flex;align-items:center;">
{dot}
<span style="font-size:0.85rem;color:{text_color};font-weight:500;">{cond}</span>
</div>"""

        st.markdown(f"""
<div class="glass-card-muted">
<div class="card-label">{selected_crop} · {len(conditions)} class{'es' if len(conditions) > 1 else ''}</div>
{cards_html}
</div>
""", unsafe_allow_html=True)

    # Data cleaning note
    st.markdown('<div class="section-title" style="margin-top:2rem;">Data Cleaning</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card-muted">
        <div class="info-text">
            After viewing and researching the dataset from GitHub, we found that we only needed one folder named
            <strong style="color:#5ecf7a;">"raw"</strong> and a subfolder named <strong style="color:#5ecf7a;">"color"</strong>.
            We used AI to generate the code that specifically pulls only that folder to our notebook workspace from GitHub
            using sparse checkout.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # EDA note
    st.markdown('<div class="section-title" style="margin-top:2rem;">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card-muted">
        <div class="info-text">
            We noticed that pictures from our dataset had minute differences — some in color and some in location relative to the leaf.
            <br><br>
            This made us understand that we could <strong>not</strong> flatten the pictures to 1D, and that we could <strong>not</strong>
            process the pictures in grayscale. We needed to preserve the full spatial and color information for the CNN to learn from.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Architecture
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠  Model Architecture":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">Sequential · 11.17M Parameters</div>
        <h1>Model <em>Architecture</em></h1>
        <p class="hero-sub">A classic CNN built from scratch with TensorFlow and Keras</p>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    # Architecture overview
    st.markdown('<div class="section-title">Design Philosophy</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="info-text">
            We structured the model using <strong style="color:#5ecf7a;">three extraction blocks</strong>, each with a
            Conv2D layer followed by MaxPooling2D. The blocks learn progressively complex features:
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Three blocks
    blocks = [
        ("Block 1 · Low-Level Features", "32 filters, 3×3 kernel", "Extracts straight edges, simple boundaries and basic shapes from the leaf images."),
        ("Block 2 · Mid-Level Features", "64 filters, 3×3 kernel", "Extracts curves, textures, and more complex patterns like vein structures."),
        ("Block 3 · High-Level Features", "128 filters, 3×3 kernel", "Extracts complex features like the actual disease spots, lesions, and discoloration patterns."),
    ]
    for title, config, desc in blocks:
        st.markdown(f"""
        <div class="glass-card-muted">
            <div class="card-label">{title}</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#5ecf7a;margin-bottom:0.4rem;">{config}</div>
            <div class="info-text">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # Layer table
    st.markdown('<div class="section-title" style="margin-top:2rem;">Layer Summary</div>', unsafe_allow_html=True)

    layers = [
        ("Conv2D",       "(None, 222, 222, 32)",  "896"),
        ("MaxPooling2D", "(None, 111, 111, 32)",  "0"),
        ("Conv2D",       "(None, 109, 109, 64)",  "18,496"),
        ("MaxPooling2D", "(None, 54, 54, 64)",    "0"),
        ("Conv2D",       "(None, 52, 52, 128)",   "73,856"),
        ("MaxPooling2D", "(None, 26, 26, 128)",   "0"),
        ("Flatten",      "(None, 86528)",          "0"),
        ("Dense",        "(None, 128)",            "11,075,712"),
        ("Dropout (0.5)","(None, 128)",            "0"),
        ("Dense (softmax)","(None, 38)",           "4,902"),
    ]

    table_rows = ""
    for name, shape, params in layers:
        table_rows += f"<tr><td>{name}</td><td>{shape}</td><td>{params}</td></tr>"

    st.markdown(f"""
    <div class="glass-card">
        <table class="layer-table">
            <thead>
                <tr><th>Layer</th><th>Output Shape</th><th>Parameters</th></tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
        <div style="display:flex;justify-content:space-between;margin-top:0.8rem;padding-top:0.8rem;border-top:1px solid rgba(94,207,122,0.12);">
            <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#7a9982;">Total Parameters</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.85rem;color:#5ecf7a;font-weight:500;">11,173,862 (42.62 MB)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key decisions
    st.markdown('<div class="section-title" style="margin-top:2rem;">Key Design Decisions</div>', unsafe_allow_html=True)

    decisions = [
        ("Optimizer", "Adam", "Learning rate of 0.001 for adaptive gradient descent."),
        ("Loss Function", "Categorical Crossentropy", "Required for our one-hot encoded 38-class labels."),
        ("Dropout Rate", "50%", "Highly aggressive dropout to combat overfitting so the model does not just memorize the training data."),
        ("Activation", "Softmax (output)", "Produces probability distribution across all 38 classes."),
        ("Input Size", "224 × 224 × 3", "Standard input size for CNN models. RGB color channels preserved."),
    ]

    for title, value, desc in decisions:
        st.markdown(f"""
        <div class="glass-card-muted" style="display:flex;gap:1rem;align-items:flex-start;">
            <div style="flex-shrink:0;">
                <div class="card-label" style="margin-bottom:0.2rem;">{title}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.9rem;color:#5ecf7a;">{value}</div>
            </div>
            <div class="info-text" style="flex:1;padding-top:0.15rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # Preprocessing
    st.markdown('<div class="section-title" style="margin-top:2rem;">Preprocessing Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">Three-Stage Optimization</div>
        <div style="margin-top:0.8rem;">
            <div class="timeline-item">
                <div class="timeline-dot"></div>
                <div class="timeline-content">
                    <div class="timeline-step">Normalization</div>
                    <div class="timeline-desc">Min-Max scaling converts pixel values from 0–255 to 0–1 range using tf.keras.layers.Rescaling(1./255)</div>
                </div>
            </div>
            <div class="timeline-item">
                <div class="timeline-dot"></div>
                <div class="timeline-content">
                    <div class="timeline-step">Parallel Processing</div>
                    <div class="timeline-desc">tf.data.AUTOTUNE dynamically assigns resources to execute data loading tasks in parallel</div>
                </div>
            </div>
            <div class="timeline-item">
                <div class="timeline-dot"></div>
                <div class="timeline-content">
                    <div class="timeline-step">Cache + Prefetch</div>
                    <div class="timeline-desc">.cache().prefetch fetches batch N+1 while N loads, achieving non-sequential pipeline execution</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Training Results
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Training Results":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">17 Epochs · Best @ Epoch 12</div>
        <h1>Training <em>Results</em></h1>
        <p class="hero-sub">Visualizing model performance across the training run</p>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    # Summary stats
    best_val_acc = max(TRAINING_HISTORY['val_accuracy'])
    best_val_loss = min(TRAINING_HISTORY['val_loss'])
    best_epoch = TRAINING_HISTORY['val_loss'].index(best_val_loss) + 1
    final_train_acc = TRAINING_HISTORY['accuracy'][-1]

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-pill">
            <div class="stat-num">{best_val_acc*100:.1f}%</div>
            <div class="stat-label">Best Val Accuracy</div>
        </div>
        <div class="stat-pill">
            <div class="stat-num">{best_val_loss:.4f}</div>
            <div class="stat-label">Best Val Loss</div>
        </div>
        <div class="stat-pill">
            <div class="stat-num">{best_epoch}</div>
            <div class="stat-label">Best Epoch</div>
        </div>
        <div class="stat-pill">
            <div class="stat-num">17</div>
            <div class="stat-label">Total Epochs</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Training vs Validation Accuracy bar chart
    st.markdown('<div class="section-title" style="margin-top:2rem;">Accuracy per Epoch</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Training (green) vs Validation (teal) accuracy</div>', unsafe_allow_html=True)

    acc_html = ""
    for i in range(len(TRAINING_HISTORY['accuracy'])):
        train_a = TRAINING_HISTORY['accuracy'][i]
        val_a = TRAINING_HISTORY['val_accuracy'][i]
        highlight = " opacity:1;" if (i + 1) == best_epoch else " opacity:0.7;"
        acc_html += f"""
<div style="margin-bottom:0.5rem;{highlight}">
<div class="epoch-row">
<span class="epoch-label">{i+1}</span>
<div class="epoch-bar-bg">
<div class="epoch-bar-fill" style="width:{train_a*100:.1f}%;background:linear-gradient(90deg,#1a6b30,#2a8a45);"></div>
</div>
<span class="epoch-val">{train_a*100:.1f}%</span>
</div>
<div class="epoch-row" style="margin-top:-0.1rem;">
<span class="epoch-label"></span>
<div class="epoch-bar-bg">
<div class="epoch-bar-fill" style="width:{val_a*100:.1f}%;background:linear-gradient(90deg,#2a7a6a,#5ecfb0);"></div>
</div>
<span class="epoch-val" style="color:#5ecfb0;">{val_a*100:.1f}%</span>
</div>
</div>"""

    st.markdown(f"""
<div class="glass-card">
<div style="display:flex;gap:1.5rem;margin-bottom:1rem;">
<div style="display:flex;align-items:center;gap:0.4rem;">
<div style="width:12px;height:4px;border-radius:2px;background:#2a8a45;"></div>
<span style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#7a9982;letter-spacing:0.08em;">TRAINING</span>
</div>
<div style="display:flex;align-items:center;gap:0.4rem;">
<div style="width:12px;height:4px;border-radius:2px;background:#5ecfb0;"></div>
<span style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#7a9982;letter-spacing:0.08em;">VALIDATION</span>
</div>
</div>
{acc_html}
</div>
""", unsafe_allow_html=True)

    # Loss chart
    st.markdown('<div class="section-title" style="margin-top:2rem;">Loss per Epoch</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Training (green) vs Validation (teal) loss — lower is better</div>', unsafe_allow_html=True)

    max_loss = max(max(TRAINING_HISTORY['loss']), max(TRAINING_HISTORY['val_loss']))
    loss_html = ""
    for i in range(len(TRAINING_HISTORY['loss'])):
        train_l = TRAINING_HISTORY['loss'][i]
        val_l = TRAINING_HISTORY['val_loss'][i]
        highlight = " opacity:1;" if (i + 1) == best_epoch else " opacity:0.7;"
        loss_html += f"""
<div style="margin-bottom:0.5rem;{highlight}">
<div class="epoch-row">
<span class="epoch-label">{i+1}</span>
<div class="epoch-bar-bg">
<div class="epoch-bar-fill" style="width:{(train_l/max_loss)*100:.1f}%;background:linear-gradient(90deg,#1a6b30,#2a8a45);"></div>
</div>
<span class="epoch-val">{train_l:.4f}</span>
</div>
<div class="epoch-row" style="margin-top:-0.1rem;">
<span class="epoch-label"></span>
<div class="epoch-bar-bg">
<div class="epoch-bar-fill" style="width:{(val_l/max_loss)*100:.1f}%;background:linear-gradient(90deg,#2a7a6a,#5ecfb0);"></div>
</div>
<span class="epoch-val" style="color:#5ecfb0;">{val_l:.4f}</span>
</div>
</div>"""

    st.markdown(f"""
<div class="glass-card">
<div style="display:flex;gap:1.5rem;margin-bottom:1rem;">
<div style="display:flex;align-items:center;gap:0.4rem;">
<div style="width:12px;height:4px;border-radius:2px;background:#2a8a45;"></div>
<span style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#7a9982;letter-spacing:0.08em;">TRAINING</span>
</div>
<div style="display:flex;align-items:center;gap:0.4rem;">
<div style="width:12px;height:4px;border-radius:2px;background:#5ecfb0;"></div>
<span style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#7a9982;letter-spacing:0.08em;">VALIDATION</span>
</div>
</div>
{loss_html}
</div>
""", unsafe_allow_html=True)

    # Analysis
    st.markdown('<div class="section-title" style="margin-top:2rem;">Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="info-text">
            The model trained for <strong style="color:#5ecf7a;">17 epochs</strong> before EarlyStopping kicked in and
            restored the best weights from <strong style="color:#5ecf7a;">Epoch 12</strong>.
            <br><br>
            At Epoch 1, training accuracy started at about 42% while validation accuracy was around 70%.
            By Epoch 12, training accuracy reached ~88% and validation accuracy peaked at 92.39% with a validation loss of 0.2640.
            <br><br>
            After Epoch 12, the validation loss started fluctuating upward even as training accuracy kept improving —
            the classic sign of <strong style="color:#e87c5a;">overfitting</strong>, which is exactly what we expected EarlyStopping to catch.
            <br><br>
            The gap between training and validation accuracy remained relatively small throughout,
            confirming the model generalized well thanks to the 50% Dropout layer.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Training config
    st.markdown('<div class="section-title" style="margin-top:2rem;">Training Configuration</div>', unsafe_allow_html=True)

    config_items = [
        ("Train / Val Split", "80% / 20%", "43,444 training · 10,861 validation"),
        ("Batch Size", "32", "Standard mini-batch gradient descent"),
        ("Max Epochs", "50", "With EarlyStopping patience of 5"),
        ("Callbacks", "EarlyStopping + ModelCheckpoint", "Monitors val_loss and val_accuracy respectively"),
    ]

    for label, value, note in config_items:
        st.markdown(f"""
        <div class="glass-card-muted" style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div class="card-label" style="margin-bottom:0.15rem;">{label}</div>
                <div style="font-size:0.75rem;color:#5a7e63;">{note}</div>
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.95rem;color:#5ecf7a;flex-shrink:0;margin-left:1rem;">{value}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Live Classifier
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬  Live Classifier":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">Live Inference · Upload & Classify</div>
        <h1>Leaf <em>Scanner</em></h1>
        <p class="hero-sub">Upload a leaf image — the model identifies crop & disease in seconds</p>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown('<p class="section-label">Leaf Image Input</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop a leaf image here, or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

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

        # Result cards
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

        # Disease info
        if not healthy:
            info_text = DISEASE_INFO.get(condition, "")
            if not info_text:
                # Try partial match
                for key, val in DISEASE_INFO.items():
                    if key.lower() in condition.lower() or condition.lower() in key.lower():
                        info_text = val
                        break
            if info_text:
                st.markdown(f"""
                <div class="disease-info">
                    <div class="info-title">ⓘ About This Disease</div>
                    <div class="info-text">{info_text}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="disease-info-healthy">
                <div class="info-title-healthy">✓ Plant Health Status</div>
                <div class="info-text">This {crop.lower()} leaf appears healthy with no visible signs of disease or pest damage.
                Continue monitoring regularly for early detection of any issues.</div>
            </div>
            """, unsafe_allow_html=True)

        # Top-5 predictions
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

    else:
        # Show placeholder when no image uploaded
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:3rem 2rem;">
            <div style="font-size:3rem;margin-bottom:0.8rem;">🍃</div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#7a9982;margin-bottom:0.4rem;">
                Waiting for a leaf image
            </div>
            <div style="font-size:0.78rem;color:#3d5e46;">
                Upload a JPG or PNG image of a plant leaf above to start classification
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Supported crops
        st.markdown('<div class="section-title" style="margin-top:2rem;">Supported Crops</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">The model can identify diseases in these 14 crops</div>', unsafe_allow_html=True)

        crop_icons = {
            'Apple': '🍎', 'Blueberry': '🫐', 'Cherry (including sour)': '🍒',
            'Corn (maize)': '🌽', 'Grape': '🍇', 'Orange': '🍊',
            'Peach': '🍑', 'Pepper, bell': '🫑', 'Potato': '🥔',
            'Raspberry': '🫐', 'Soybean': '🫘', 'Squash': '🎃',
            'Strawberry': '🍓', 'Tomato': '🍅',
        }

        grid_html = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(100px,1fr));gap:0.6rem;">'
        for crop in UNIQUE_CROPS:
            icon = crop_icons.get(crop, '🌱')
            n_diseases = len(CROP_DISEASES[crop])
            grid_html += f"""
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(94,207,122,0.10);
                        border-radius:10px;padding:0.8rem 0.5rem;text-align:center;">
                <div style="font-size:1.4rem;">{icon}</div>
                <div style="font-size:0.7rem;color:#a8c8b0;margin-top:0.3rem;">{crop}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3d5e46;margin-top:0.1rem;">{n_diseases} class{'es' if n_diseases > 1 else ''}</div>
            </div>"""
        grid_html += '</div>'
        st.markdown(grid_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Team
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥  Team":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">TechCrush · Capstone · Group 6</div>
        <h1>The <em>Team</em></h1>
        <p class="hero-sub">C6 – AI/ML Track, Class-A</p>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    # Project info
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">About This Project</div>
        <div class="info-text">
            This capstone project was built as part of the <strong style="color:#5ecf7a;">TechCrush AI/ML Track</strong>.
            Our team collaborated to build a plant disease classification system using deep learning,
            covering the full pipeline from data acquisition to a deployed Streamlit demo.
            <br><br>
            All code was developed collaboratively with AI assistance for code generation, while the research,
            experimentation, and analysis were driven by the team.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tech stack
    st.markdown('<div class="section-title" style="margin-top:2rem;">Tech Stack</div>', unsafe_allow_html=True)

    tech = [
        ("🐍", "Python", "Core Language"),
        ("🧠", "TensorFlow / Keras", "Model Building"),
        ("📊", "Matplotlib / NumPy", "Visualization & Compute"),
        ("🖥️", "Streamlit", "Demo Application"),
        ("🌐", "Google Colab", "Training Environment"),
        ("📂", "GitHub", "Version Control"),
    ]

    tech_html = '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.6rem;margin:1rem 0;">'
    for icon, name, role in tech:
        tech_html += f"""
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(94,207,122,0.10);
                    border-radius:10px;padding:0.9rem 0.6rem;text-align:center;">
            <div style="font-size:1.3rem;">{icon}</div>
            <div style="font-size:0.78rem;color:#f0f5f1;margin-top:0.3rem;font-weight:500;">{name}</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.52rem;color:#3d5e46;letter-spacing:0.08em;margin-top:0.15rem;">{role}</div>
        </div>"""
    tech_html += '</div>'
    st.markdown(tech_html, unsafe_allow_html=True)

    # References
    st.markdown('<div class="section-title" style="margin-top:2rem;">References</div>', unsafe_allow_html=True)

    refs = [
        ("PlantVillage Dataset", "github.com/spMohanty/PlantVillage-Dataset", "https://github.com/spMohanty/PlantVillage-Dataset/"),
        ("TensorFlow Documentation", "tensorflow.org/api_docs", "https://www.tensorflow.org/api_docs"),
        ("Hughes & Salathé (2015)", "Original PlantVillage paper", ""),
    ]

    for title, subtitle, url in refs:
        link_html = f'<a href="{url}" target="_blank" style="color:#5ecf7a;text-decoration:none;font-size:0.72rem;">{subtitle} ↗</a>' if url else f'<span style="font-size:0.72rem;color:#5a7e63;">{subtitle}</span>'
        st.markdown(f"""
        <div class="glass-card-muted" style="display:flex;justify-content:space-between;align-items:center;">
            <div style="font-size:0.85rem;color:#c0d4c5;">{title}</div>
            {link_html}
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Footer (all pages)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-footer">
    <div class="footer-text">PhytoScan · Group 6 · TechCrush Capstone · AI/ML Track Class-A</div>
</div>
""", unsafe_allow_html=True)