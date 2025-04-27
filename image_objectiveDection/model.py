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

        # --- 下采样特征提取部分 ---
        self.layer1 = ConvBlock(3, 32, 3, 2)    # 640 -> 320
        self.layer2 = ConvBlock(32, 64, 3, 2)   # 320 -> 160
        self.layer3 = ConvBlock(64, 128, 3, 2)  # 160 -> 80
        self.layer4 = ConvBlock(128, 256, 3, 2) # 80 -> 40
        self.layer5 = ConvBlock(256, 512, 3, 2) # 40 -> 20

        # --- Detection head ---
        self.detect = nn.Conv2d(512, (5 + num_classes) * 3, 1)  # 3个anchor每个输出(5+num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.detect(x)

        # x shape: (batch, (5+num_classes)*3, 20, 20)
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 5 + self.num_classes, 20, 20)
        x = x.permute(0, 1, 3, 4, 2)  # (batch, 3 anchors, 20, 20, 5+num_classes)

        x = x.reshape(batch_size, -1, 5 + self.num_classes)  # (batch, N, 5+num_classes)，N=3x20x20=1200

        return x
