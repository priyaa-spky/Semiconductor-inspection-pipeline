import os
import io
import time
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import base64

app = FastAPI(title="SpectraRestore AI Engine")

# Enable CORS for local web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. U-NET ARCHITECTURE ---
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

# Load model weights
device = torch.device("cpu")
model = RestorationUNet().to(device)
weights_path = "models/restoration_unet.pth"
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

def to_base64(img_np):
    """Converts numpy image to base64 JPEG/PNG string"""
    if len(img_np.shape) == 2:
        img = Image.fromarray(img_np)
    else:
        img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/api/restore")
async def restore_micrograph(file: UploadFile = File(...), threshold: int = Form(35)):
    contents = await file.read()
    fname = file.filename
    
    # Process npy or standard image
    if fname.endswith(".npy"):
        data = np.load(io.BytesIO(contents))
        if data.ndim == 3:
            data = data[0]
        d_min, d_max = float(data.min()), float(data.max())
        norm = (data - d_min) / (d_max - d_min + 1e-6)
        img = Image.fromarray((norm * 255).astype(np.uint8))
    else:
        img = Image.open(io.BytesIO(contents)).convert("L")

    img = img.resize((128, 128))
    np_arr = np.array(img, dtype=np.float32) / 255.0
    inp_tensor = torch.from_numpy(np_arr).unsqueeze(0).unsqueeze(0)

    t0 = time.time()
    with torch.no_grad():
        out_tensor = model(inp_tensor)
    latency_ms = (time.time() - t0) * 1000

    out_np = (out_tensor.squeeze().numpy() * 255.0).astype(np.uint8)
    orig_np = (np_arr * 255.0).astype(np.uint8)

    # Metrics
    mse = np.mean((orig_np.astype(np.float64) - out_np.astype(np.float64)) ** 2)
    psnr = 20 * np.log10(255.0 / (np.sqrt(mse) + 1e-6)) if mse > 0 else 100.0

    # Residuals & Defect Segmentation
    diff = cv2.absdiff(out_np, orig_np)
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_VIRIDIS)

    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated = cv2.cvtColor(out_np, cv2.COLOR_GRAY2BGR)
    defect_count = sum(1 for c in contours if cv2.contourArea(c) >= 4)
    for c in contours:
        if cv2.contourArea(c) >= 4:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (234, 67, 53), 1)

    return JSONResponse({
        "status": "FLAGGED (Defects Detected)" if defect_count > 0 else "PASS (Clean Wafer)",
        "is_defective": bool(defect_count > 0),
        "psnr": round(float(psnr), 2),
        "defects": int(defect_count),
        "latency_ms": round(float(latency_ms), 1),
        "raw_img": to_base64(orig_np),
        "restored_img": to_base64(out_np),
        "heatmap_img": to_base64(heatmap),
        "annotated_img": to_base64(annotated)
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)