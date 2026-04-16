
"""
DualForensics: Dual-branch spatiotemporal attention network.

"""

import torch
import torch.nn as nn
from configs.settings import FEATURE_DIM, DROPOUT, DEVICE
from src.models.backbone import EfficientNetBackbone
from src.models.attention import CBAM
from src.models.temporal import TemporalTransformer
from src.models.fusion import CrossAttentionFusion


class DualForensics(nn.Module):
    """
    Dual-branch architecture:
      Spatial branch: EfficientNet + CBAM
      Temporal branch: EfficientNet + Transformer
      Fusion: Cross-attention
    """

    def __init__(self):
        super().__init__()
        D = FEATURE_DIM

        self.backbone = EfficientNetBackbone(pretrained=True)
        self.spatial_attention = CBAM(D)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.temporal_transformer = TemporalTransformer(D)
        self.fusion = CrossAttentionFusion(D, D)

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion.output_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        B, N, C, H, W = x.shape
        flat = x.view(B * N, C, H, W)

        fmaps, fvecs = self.backbone(flat)
        D = fvecs.size(1)
        h, w = fmaps.shape[2], fmaps.shape[3]

        attended, s_maps = self.spatial_attention(fmaps)
        spatial_feats = self.spatial_pool(attended).flatten(1).view(B, N, D)

        temporal_in = fvecs.view(B, N, D)
        temporal_feats, t_attn = self.temporal_transformer(temporal_in)

        fused, c_attn = self.fusion(spatial_feats, temporal_feats)
        logits = self.classifier(fused)

        explain = {
            "spatial_maps": s_maps.view(B, N, 1, h, w).detach(),
            "temporal_attn": t_attn.detach(),
            "cross_attn": c_attn.detach(),
        }
        return logits, explain


class CNNBaseline(nn.Module):
    """Single-frame CNN baseline. Averages frame-level predictions."""

    def __init__(self):
        super().__init__()
        self.backbone = EfficientNetBackbone()
        self.classifier = nn.Sequential(
            nn.Linear(FEATURE_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        B, N, C, H, W = x.shape
        _, vecs = self.backbone(x.view(B * N, C, H, W))
        logits = self.classifier(vecs).view(B, N, 1).mean(dim=1)
        return logits, {}


class CNNLSTMBaseline(nn.Module):
    """CNN + bidirectional LSTM baseline."""

    def __init__(self, hidden=256, layers=2):
        super().__init__()
        self.backbone = EfficientNetBackbone()
        self.lstm = nn.LSTM(
            FEATURE_DIM, hidden, layers,
            batch_first=True, dropout=0.3, bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        B, N, C, H, W = x.shape
        _, vecs = self.backbone(x.view(B * N, C, H, W))
        out, _ = self.lstm(vecs.view(B, N, -1))
        logits = self.classifier(out[:, -1, :])
        return logits, {}


def build_model(name="dualforensics"):
    models = {
        "dualforensics": DualForensics,
        "cnn_only": CNNBaseline,
        "cnn_lstm": CNNLSTMBaseline,
    }
    model = models[name]().to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model '{name}': {total:,} params ({trainable:,} trainable)")
    return model








