"""U-Net with an ImageNet-pretrained ResNet-34 encoder. Outputs one logit per pixel (crack or not)."""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34


class Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class CrackUNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        enc = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)       # 1/2, 64
        self.pool = enc.maxpool
        self.l1, self.l2, self.l3, self.l4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4  # 1/4..1/32
        self.up4 = Up(512, 256, 256)
        self.up3 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up1 = Up(64, 64, 48)
        # full-resolution skip so hairline cracks survive
        self.full = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.up0 = Up(48, 16, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        s0 = self.full(x)
        s1 = self.stem(x)
        s2 = self.l1(self.pool(s1))
        s3 = self.l2(s2)
        s4 = self.l3(s3)
        s5 = self.l4(s4)
        d = self.up4(s5, s4)
        d = self.up3(d, s3)
        d = self.up2(d, s2)
        d = self.up1(d, s1)
        d = self.up0(d, s0)
        return self.head(d)
