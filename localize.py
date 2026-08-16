import argparse
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

# U-Net Architecture
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

def load_and_restore(img_path, model, target_shape=(128, 128)):
    if img_path.endswith(".npy"):
        data = np.load(img_path)
        if data.ndim == 3:
            data = data[0]
        d_min, d_max = float(data.min()), float(data.max())
        norm = (data - d_min) / (d_max - d_min + 1e-6)
        img = (norm * 255).astype(np.uint8)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    orig_h, orig_w = img.shape
    resized = cv2.resize(img, target_shape)
    inp = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        restored = model(inp)
    
    res_np = (restored.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
    restored_original_scale = cv2.resize(res_np, (orig_w, orig_h))
    return restored_original_scale

def locate_pattern(ref_img, search_img):
    # Normalized Cross Correlation on restored features
    result = cv2.matchTemplate(search_img, ref_img, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    
    ref_h, ref_w = ref_img.shape
    center_x = max_loc[0] + (ref_w // 2)
    center_y = max_loc[1] + (ref_h // 2)
    return center_x, center_y

def main():
    parser = argparse.ArgumentParser(description="Semiconductor Pattern Localization Inference")
    parser.add_argument("--ref", type=str, required=True, help="Path to reference pattern image")
    parser.add_argument("--search", type=str, required=True, help="Path to search scene image")
    parser.add_argument("--weights", type=str, default="models/restoration_unet.pth", help="Path to trained model weights")
    args = parser.parse_args()

    if not os.path.exists(args.ref) or not os.path.exists(args.search):
        print("Error: Input files not found.", file=sys.stderr)
        sys.exit(1)

    model = RestorationUNet()
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    ref_restored = load_and_restore(args.ref, model)
    search_restored = load_and_restore(args.search, model)

    pred_x, pred_y = locate_pattern(ref_restored, search_restored)

    # Standard stdout output for automated evaluation runners
    print(f"Predicted Center: ({pred_x}, {pred_y})")
    print(f"PRED_X={pred_x}")
    print(f"PRED_Y={pred_y}")

if __name__ == "__main__":
    main()