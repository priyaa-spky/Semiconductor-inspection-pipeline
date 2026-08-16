import os
import glob
import zipfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# --- 1. U-NET MODEL ---
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

# --- 2. FAST CPU DATASET ---
class FastNpyDataset(Dataset):
    def __init__(self, root_dir, max_samples=1200, size=128):
        self.size = size
        self.npy_paths = glob.glob(os.path.join(root_dir, "**", "*.npy"), recursive=True)[:max_samples]

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        data = np.load(self.npy_paths[idx])
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim == 3:
            data = data[0:1, :, :]

        min_v, max_v = data.min(), data.max()
        data = (data - min_v) / (max_v - min_v + 1e-6)
        
        clean = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
        clean = F.interpolate(clean, size=(self.size, self.size), mode='bilinear', align_corners=False)
        
        lr = F.interpolate(clean, size=(self.size // 2, self.size // 2), mode='bilinear', align_corners=False)
        noisy = torch.clamp(lr + lr * torch.randn_like(lr) * 0.1, 0.0, 1.0)
        degraded = F.interpolate(noisy, size=(self.size, self.size), mode='bilinear', align_corners=False)
        
        return degraded.squeeze(0), clean.squeeze(0)

# --- 3. TRAINING EXECUTION ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on: {device}")

    # Uses 1,200 samples & 128x128 resolution for lightning CPU execution
    dataset = FastNpyDataset(".", max_samples=1200, size=128)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = RestorationUNet().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    os.makedirs("models", exist_ok=True)
    EPOCHS = 3

    print(f"[*] Training started: {len(dataset)} samples | {len(loader)} steps per epoch...")
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        model.train()
        for step, (inputs, targets) in enumerate(loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if step % 10 == 0 or step == len(loader):
                print(f"Epoch [{epoch}/{EPOCHS}] | Step [{step}/{len(loader)}] | Current Loss: {loss.item():.5f}")
                
        print(f"--> Epoch {epoch} Completed | Avg Loss: {total_loss / len(loader):.5f}\n")

    torch.save(model.state_dict(), "models/restoration_unet.pth")
    print("[*] Complete! Weights saved to: models/restoration_unet.pth")