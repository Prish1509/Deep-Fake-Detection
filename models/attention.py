
"""
CBAM: Convolutional Block Attention Module.
Channel attention learns which features matter.
Spatial attention learns where to look.
"""

import torch
import torch.nn as nn
from configs.settings import SPATIAL_ATT_REDUCTION


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=SPATIAL_ATT_REDUCTION):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * att


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        att_map = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att_map, att_map


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x, s_map = self.spatial_att(x)
        return x, s_map




