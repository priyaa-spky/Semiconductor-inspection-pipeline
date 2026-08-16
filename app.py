import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import os
import io
import time
import pandas as pd

# Set Page Config
st.set_page_config(
    page_title="SpectraRestore Studio | Semiconductor QA",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- COHESIVE WHITE & LAVENDER THEME OVERRIDES ---
st.markdown("""
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" />

    <style>
        /* Global Typography & Palette */
        html, body, [class*="css"], * {
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        }

        /* 1. Remove Streamlit default black header */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
            height: 0.5rem !important;
        }

        /* 2. Main Page Background: Light lavender canvas */
        .stApp {
            background-color: #faf7ff !important;
            color: #1e1b4b !important;
        }

        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            max-width: 1240px;
        }

        /* Top Brand Navigation Bar */
        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 14px;
            border-bottom: 1.5px solid #ede9fe;
            margin-bottom: 22px;
        }
        .brand-title {
            font-size: 1.45rem;
            font-weight: 700;
            color: #1e1b4b;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .brand-pill {
            background: #ede9fe;
            color: #6d28d9;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 6px 14px;
            border-radius: 9999px;
            border: 1px solid #ddd6fe;
        }

        /* White Surface Cards */
        .lavender-card {
            background-color: #ffffff !important;
            border: 1.5px solid #ede9fe !important;
            border-radius: 16px !important;
            padding: 24px !important;
            box-shadow: 0 4px 20px -2px rgba(124, 58, 237, 0.06) !important;
        }

        /* --- DRAG & DROP UPLOADER (LIGHT LAVENDER THEME) --- */
        div[data-testid="stFileUploader"] {
            background: transparent !important;
        }
        div[data-testid="stFileUploaderDropzone"] {
            background-color: #f5f0ff !important;
            border: 2px dashed #c4b5fd !important;
            border-radius: 14px !important;
            padding: 28px 18px !important;
            transition: all 0.2s ease-in-out !important;
        }
        div[data-testid="stFileUploaderDropzone"]:hover {
            background-color: #ede5ff !important;
            border-color: #8b5cf6 !important;
        }
        div[data-testid="stFileUploaderDropzone"] span,
        div[data-testid="stFileUploaderDropzone"] small,
        div[data-testid="stFileUploaderDropzone"] div {
            color: #5b21b6 !important;
            font-weight: 500 !important;
        }
        div[data-testid="stFileUploaderDropzone"] svg {
            fill: #7c3aed !important;
            color: #7c3aed !important;
            width: 32px !important;
            height: 32px !important;
        }
        div[data-testid="stFileUploaderDropzone"] button {
            background-color: #7c3aed !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 20px !important;
            padding: 6px 20px !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 8px rgba(124, 58, 237, 0.25) !important;
        }
        div[data-testid="stFileUploaderDropzone"] button:hover {
            background-color: #6d28d9 !important;
        }

        /* Uploaded File Pill */
        div[data-testid="stFileUploaderFile"] {
            background-color: #ede9fe !important;
            border: 1px solid #c4b5fd !important;
            border-radius: 10px !important;
            margin-top: 10px !important;
        }
        div[data-testid="stFileUploaderFile"] * {
            color: #4c1d95 !important;
            font-weight: 600 !important;
        }

        /* Slider Lavender Accent */
        div[data-testid="stSlider"] div[role="slider"] {
            background-color: #7c3aed !important;
            border-color: #7c3aed !important;
        }

        /* Metric Tiles */
        div[data-testid="stMetric"] {
            background-color: #ffffff !important;
            border: 1.5px solid #ede9fe !important;
            border-radius: 12px !important;
            padding: 16px 20px !important;
            box-shadow: 0 2px 10px rgba(124, 58, 237, 0.04) !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
        }
        div[data-testid="stMetricValue"] {
            color: #6d28d9 !important;
            font-size: 1.45rem !important;
            font-weight: 700 !important;
        }

        /* Lavender Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            border-bottom: 1.5px solid #ede9fe;
            background: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #64748b;
            font-size: 0.95rem;
            font-weight: 600;
            padding: 10px 18px;
            border-radius: 8px 8px 0 0;
        }
        .stTabs [aria-selected="true"] {
            color: #7c3aed !important;
            border-bottom: 3px solid #7c3aed !important;
            background: transparent !important;
        }

        /* Action Buttons */
        .stButton>button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
            color: #ffffff !important;
            border: none !important;
            height: 2.7rem !important;
            box-shadow: 0 3px 12px rgba(124, 58, 237, 0.25) !important;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #6d28d9 0%, #5b21b6 100%) !important;
        }
        .stDownloadButton>button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            background-color: #f5f0ff !important;
            color: #6d28d9 !important;
            border: 1.5px solid #ddd6fe !important;
            height: 2.7rem !important;
        }
        .stDownloadButton>button:hover {
            background-color: #ede5ff !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- 1. MODEL ARCHITECTURE ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU()
        )
    def forward(self, x):
        return self.net(x)

class RestorationUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(1, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.bot = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv2 = DoubleConv(64, 32)
        self.up3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv3 = DoubleConv(32, 16)
        self.outc = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.bot(x3)
        d1 = self.conv1(torch.cat([self.up1(x4), x3], dim=1))
        d2 = self.conv2(torch.cat([self.up2(d1), x2], dim=1))
        d3 = self.conv3(torch.cat([self.up3(d2), x1], dim=1))
        return torch.clamp(self.outc(d3) + x, 0.0, 1.0)

# --- 2. LOAD MODEL WEIGHTS ---
@st.cache_resource
def get_model():
    m = RestorationUNet()
    weights_path = "models/restoration_unet.pth"
    if os.path.exists(weights_path):
        m.load_state_dict(torch.load(weights_path, map_location="cpu"))
    m.eval()
    return m

model = get_model()

# --- 3. INFERENCE PIPELINE ---
def process_micrograph(raw_bytes, filename, threshold_val=35):
    if filename.endswith(".npy"):
        data = np.load(io.BytesIO(raw_bytes))
        if data.ndim == 3:
            data = data[0]
        d_min, d_max = float(data.min()), float(data.max())
        norm = (data - d_min) / (d_max - d_min + 1e-6)
        img = Image.fromarray((norm * 255).astype(np.uint8))
    else:
        img = Image.open(io.BytesIO(raw_bytes)).convert("L")

    img = img.resize((128, 128))
    np_arr = np.array(img, dtype=np.float32) / 255.0
    inp_tensor = torch.from_numpy(np_arr).unsqueeze(0).unsqueeze(0)

    t0 = time.time()
    with torch.no_grad():
        out_tensor = model(inp_tensor)
    latency_ms = (time.time() - t0) * 1000

    out_np = (out_tensor.squeeze().numpy() * 255.0).astype(np.uint8)
    orig_np = (np_arr * 255.0).astype(np.uint8)

    mse = np.mean((orig_np.astype(np.float64) - out_np.astype(np.float64)) ** 2)
    psnr = 20 * np.log10(255.0 / (np.sqrt(mse) + 1e-6)) if mse > 0 else 100.0

    diff = cv2.absdiff(out_np, orig_np)
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_VIRIDIS)

    _, mask = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated = cv2.cvtColor(out_np, cv2.COLOR_GRAY2BGR)
    defect_count = sum(1 for c in contours if cv2.contourArea(c) >= 4)
    for c in contours:
        if cv2.contourArea(c) >= 4:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (234, 67, 53), 1)

    return {
        "filename": filename,
        "original": orig_np,
        "restored": out_np,
        "heatmap": heatmap,
        "annotated": annotated,
        "psnr": psnr,
        "latency_ms": latency_ms,
        "defect_count": defect_count,
        "status": "FLAGGED" if defect_count > 0 else "PASS"
    }

# --- 4. TOP APP BAR WITH EMBEDDED NATIVE SVG ICON ---
st.markdown("""
    <div class="app-header">
        <div class="brand-title">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="2" x2="12" y2="6"></line>
                <line x1="12" y1="18" x2="12" y2="22"></line>
                <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line>
                <line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line>
                <line x1="2" y1="12" x2="6" y2="12"></line>
                <line x1="18" y1="12" x2="22" y2="12"></line>
                <line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line>
                <line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line>
            </svg>
            <span>SpectraRestore Studio <span style="font-size:0.95rem; font-weight:400; color:#64748b;">| Silicon Defect & Reconstruction Engine</span></span>
        </div>
        <div class="brand-pill">PyTorch Residual U-Net</div>
    </div>
""", unsafe_allow_html=True)

# Navigation Tabs
tabs = st.tabs(["Micrograph Inspection", "Disk Batch Evaluation (940MB+)", "Model Specifications"])

# ================= TAB 1: INSPECTION (INPUT ON RIGHT, VIEWPORTS ON LEFT) =================
with tabs[0]:
    col_display, col_controls = st.columns([2.7, 1], gap="large")

    # RIGHT COLUMN: Input Box & Boundary Sensitivity
    with col_controls:
        # Large Input Micrograph Header
        st.markdown("""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
                <span style="font-weight:700; color:#1e1b4b; font-size:1.05rem;">Input Micrograph</span>
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload .npy or image", 
            type=["npy", "png", "jpg", "jpeg"], 
            label_visibility="collapsed",
            key="primary_micrograph_uploader"
        )

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        
        # Large Boundary Sensitivity Header
        st.markdown("""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="4" y1="21" x2="4" y2="14"></line>
                    <line x1="4" y1="10" x2="4" y2="3"></line>
                    <line x1="12" y1="21" x2="12" y2="12"></line>
                    <line x1="12" y1="8" x2="12" y2="3"></line>
                    <line x1="20" y1="21" x2="20" y2="16"></line>
                    <line x1="20" y1="12" x2="20" y2="3"></line>
                    <line x1="1" y1="14" x2="7" y2="14"></line>
                    <line x1="9" y1="8" x2="15" y2="8"></line>
                    <line x1="17" y1="16" x2="23" y2="16"></line>
                </svg>
                <span style="font-weight:700; color:#1e1b4b; font-size:1.05rem;">Boundary Sensitivity</span>
            </div>
        """, unsafe_allow_html=True)

        threshold_val = st.slider(
            "Sensitivity Threshold", 
            min_value=10, 
            max_value=100, 
            value=35, 
            label_visibility="collapsed",
            key="boundary_sensitivity_slider"
        )
        st.caption("Lower threshold isolates microscopic sub-surface variations on silicon tracks.")

    # LEFT COLUMN: Main Micrograph Results & Metrics
    with col_display:
        if uploaded_file is not None:
            res = process_micrograph(uploaded_file.read(), uploaded_file.name, threshold_val)
            
            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Verdict", res["status"])
            m2.metric("Estimated PSNR", f"{res['psnr']:.2f} dB")
            m3.metric("Defects Located", f"{res['defect_count']} Clusters")
            m4.metric("Inference Time", f"{res['latency_ms']:.1f} ms")
            
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            
            # 4 Viewports
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**1. Raw Input Micrograph**")
                st.image(res["original"], use_container_width=True)
                st.markdown("**3. Noise Residual Heatmap**")
                st.image(res["heatmap"], use_container_width=True)
            with g2:
                st.markdown("**2. AI Restored Output (Denoised)**")
                st.image(res["restored"], use_container_width=True)
                st.markdown("**4. Defect Bounding Matrix**")
                st.image(res["annotated"], use_container_width=True)
                
            buf = io.BytesIO()
            Image.fromarray(res["restored"]).save(buf, format="PNG")
            st.download_button("Export Restored PNG", data=buf.getvalue(), file_name=f"restored_{res['filename']}.png", mime="image/png", key="download_restored_btn")
        else:
            # High-Impact Lavender Placeholder Box with Large SVG Icon
            st.markdown("""
                <div class="lavender-card" style="text-align: center; padding: 60px 20px;">
                    <div style="width: 80px; height: 80px; margin: 0 auto 16px auto; background-color: #ede9fe; border-radius: 24px; display: flex; align-items: center; justify-content: center; border: 1.5px solid #ddd6fe;">
                        <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                            <circle cx="8.5" cy="8.5" r="1.5"></circle>
                            <polyline points="21 15 16 10 5 21"></polyline>
                        </svg>
                    </div>
                    <h3 style="color:#1e1b4b; font-weight:700; font-size:1.2rem; margin-bottom: 6px;">No Micrograph Loaded</h3>
                    <p style="color:#64748b; font-size:0.9rem; max-width:420px; margin: 0 auto;">Select a <code style="color:#7c3aed; font-weight:600;">.npy</code> scan or microscope image from the right panel to execute real-time super-resolution reconstruction.</p>
                </div>
            """, unsafe_allow_html=True)

# ================= TAB 2: BATCH EVALUATION =================
with tabs[1]:
    st.markdown("<div class='lavender-card'>", unsafe_allow_html=True)
    st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
            </svg>
            <span style="font-weight:700; color:#1e1b4b; font-size:1.15rem;">Large Dataset Sequential Batch Evaluation</span>
        </div>
    """, unsafe_allow_html=True)
    st.caption("Reads directly from disk to evaluate bulk datasets without memory pressure.")
    
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    b1, b2 = st.columns([2, 1])
    with b1:
        fpath = st.text_input("Dataset Directory Path:", value="extracted_npy", key="batch_dir_path")
    with b2:
        limit = st.number_input("Samples to Evaluate:", min_value=10, max_value=500, value=40, step=10, key="batch_sample_limit")
        
    if os.path.exists(fpath):
        all_samples = [os.path.join(r, f) for r, _, fs in os.walk(fpath) for f in fs if f.endswith(('.npy', '.png', '.jpg'))]
        st.write(f"Located **{len(all_samples)}** candidate scan files.")
        
        if st.button("Run Batch Pipeline", key="run_batch_pipeline_btn"):
            bar = st.progress(0)
            rows = []
            for i, p in enumerate(all_samples[:limit]):
                with open(p, "rb") as f:
                    out = process_micrograph(f.read(), os.path.basename(p), 35)
                rows.append({
                    "Sample": out["filename"],
                    "Status": out["status"],
                    "Defect Count": out["defect_count"],
                    "PSNR (dB)": round(out["psnr"], 2),
                    "Latency (ms)": round(out["latency_ms"], 1)
                })
                bar.progress((i + 1) / limit)
                
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Batch Report (CSV)", data=csv, file_name="qa_evaluation_report.csv", mime="text/csv", key="download_batch_csv_btn")
    else:
        st.error(f"Directory `{fpath}` not found.")
    st.markdown("</div>", unsafe_allow_html=True)

# ================= TAB 3: SPECIFICATIONS =================
with tabs[2]:
    st.markdown("<div class='lavender-card'>", unsafe_allow_html=True)
    st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect>
                <rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect>
                <line x1="6" y1="6" x2="6.01" y2="6"></line>
                <line x1="6" y1="18" x2="6.01" y2="18"></line>
            </svg>
            <span style="font-weight:700; color:#1e1b4b; font-size:1.15rem;">Model Specifications</span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
| Specifier | Value |
| :--- | :--- |
| **Model Type** | Residual U-Net (`DoubleConv` + Transposed Upsampling) |
| **Nonlinear Activation** | Parametric ReLU (`PReLU`) |
| **Input Shape** | $(1, 1, 128, 128)$ Single-Channel Micrograph |
| **Loss Function** | $\mathcal{L}_1$ Mean Absolute Error |
| **Checkpoint Path** | `models/restoration_unet.pth` |
| **Inference Device** | PyTorch CPU Engine |
    """)
    st.markdown("</div>", unsafe_allow_html=True)