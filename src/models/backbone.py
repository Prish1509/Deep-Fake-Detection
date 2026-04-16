
"""
CNN backbone for feature extraction.
"""

import torch.nn as nn
from torchvision import models


class EfficientNetBackbone(nn.Module):
    """EfficientNet-B0 pretrained on ImageNet. Returns feature maps and vector."""

    def __init__(self, pretrained=True):
        super().__init__()
        base = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 1280

    def forward(self, x):
        fmaps = self.features(x)
        fvec = self.pool(fmaps).flatten(1)
        return fmaps, fvec
