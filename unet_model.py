import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => GroupNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=out_channels // 2, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=out_channels // 2, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffZ // 2, diffZ - diffZ // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffX // 2, diffX - diffX // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3d(nn.Module):
    def __init__(self, n_channels=4, n_classes=3):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Contracting path
        self.conv = DoubleConv(n_channels, 16)
        self.enc1 = Down(16, 32)
        self.enc2 = Down(32, 64)
        self.enc3 = Down(64, 128)
        self.enc4 = Down(128, 256)

        # Expansive path
        self.dec1 = Up(256 + 128, 128)
        self.dec2 = Up(128 + 64, 64)
        self.dec3 = Up(64 + 32, 32)
        self.dec4 = Up(32 + 16, 16)
        
        self.out = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        
        x = self.dec1(x5, x4)
        x = self.dec2(x, x3)
        x = self.dec3(x, x2)
        x = self.dec4(x, x1)
        
        logits = self.out(x)
        return logits
