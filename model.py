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
        d2 = self.dec2(torch.cat([e3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        return self.final(d1)