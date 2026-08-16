import torch
import torch.nn as nn

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
        self.inc = DoubleConv(1, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.bot = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        
        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv4 = DoubleConv(64, 32)
        
        self.outc = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bot(x4)
        
        d1 = self.conv1(torch.cat([self.up1(x5), x4], dim=1))
        d2 = self.conv2(torch.cat([self.up2(d1), x3], dim=1))
        d3 = self.conv3(torch.cat([self.up3(d2), x2], dim=1))
        d4 = self.conv4(torch.cat([self.up4(d3), x1], dim=1))
        
        return torch.clamp(self.outc(d4) + x, 0.0, 1.0)