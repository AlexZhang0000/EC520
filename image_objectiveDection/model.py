# --- model_fast_improved.py ---

import torch
import torch.nn as nn

class YOLOv5Backbone(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # High-resolution head (20x20)
        self.head_high = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3 * (5 + num_classes), 1)
        )

        # Low-resolution head (10x10)
        self.downsample = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.head_low = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3 * (5 + num_classes), 1)
        )

    def forward(self, x):
        feat = self.features(x)  # (batch,128,20,20)

        out_high = self.head_high(feat)  # (batch, 3*(5+C),20,20)

        feat_low = self.downsample(feat)  # (batch,256,10,10)
        out_low = self.head_low(feat_low) # (batch, 3*(5+C),10,10)

        return [out_high, out_low]


