import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        
        # Decoder  
        self.dec2 = self._block(256 + 128, 128)  # Skip connection
        self.dec1 = self._block(128 + 64, 64)    # Skip connection
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
    
    def forward(self, x, t):
        # x: input image, t: timestep (akan kita sederhanakan)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        
        # Decoder dengan skip connections
        # UPSAMPLE e3 dari 16x16 ke 32x32 sebelum concat dengan e2
        e3_upsampled = F.interpolate(e3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([e3_upsampled, e2], dim=1))
        
        # UPSAMPLE d2 dari 32x32 ke 64x64 sebelum concat dengan e1
        d2_upsampled = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d2_upsampled, e1], dim=1))
        
        return self.final(d1)
