import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class YOLOv5Backbone(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # Backbone: downsample + feature extraction
        self.layer1 = ConvBlock(3, 32, 3, 2)    # 640 -> 320
        self.layer2 = ConvBlock(32, 64, 3, 2)   # 320 -> 160
        self.layer3 = ConvBlock(64, 128, 3, 2)  # 160 -> 80
        self.layer4 = ConvBlock(128, 256, 3, 2) # 80 -> 40
        self.layer5 = ConvBlock(256, 512, 3, 2) # 40 -> 20
        self.layer6 = ConvBlock(512, 512, 3, 2) # 20 -> 10

        # FPN Neck: upsample + fusion
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse1 = ConvBlock(512 + 512, 512)
        self.fuse2 = ConvBlock(512 + 256, 256)

        # Final detection head
        self.detect = nn.Conv2d(256, (5 + num_classes) * 3, 1)  # 3 anchors per position

    def forward(self, x):
        x1 = self.layer1(x)  # 320
        x2 = self.layer2(x1) # 160
        x3 = self.layer3(x2) # 80
        x4 = self.layer4(x3) # 40
        x5 = self.layer5(x4) # 20
        x6 = self.layer6(x5) # 10

        # FPN
        p5 = self.upsample(x6)             # 10 -> 20
        p5 = torch.cat([p5, x5], dim=1)     # concat with 20x20
        p5 = self.fuse1(p5)

        p4 = self.upsample(p5)              # 20 -> 40
        p4 = torch.cat([p4, x4], dim=1)     # concat with 40x40
        p4 = self.fuse2(p4)

        out = self.detect(p4)               # output on 40x40

        batch_size = out.size(0)
        out = out.view(batch_size, 3, 5 + self.num_classes, 40, 40)
        out = out.permute(0, 1, 3, 4, 2)  # (batch, 3 anchors, 40, 40, 5+num_classes)
        out = out.reshape(batch_size, -1, 5 + self.num_classes)  # (batch, N, 5+num_classes), N = 3*40*40=4800

        return out

