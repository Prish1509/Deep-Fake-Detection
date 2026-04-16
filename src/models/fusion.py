
"""
Cross-attention fusion: spatial queries attend to temporal keys/values.

"""

import torch.nn as nn
from configs.settings import FUSION_DIM


class CrossAttentionFusion(nn.Module):
    def __init__(self, spatial_dim, temporal_dim, fusion_dim=FUSION_DIM):
        super().__init__()
        self.s_proj = nn.Linear(spatial_dim, fusion_dim)
        self.t_proj = nn.Linear(temporal_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=4,
            dropout=0.1, batch_first=True,
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.output_dim = fusion_dim

    def forward(self, spatial_feats, temporal_feats):
        s = self.s_proj(spatial_feats)
        t = self.t_proj(temporal_feats)
        attn_out, attn_w = self.cross_attn(s, t, t)
        fused = self.norm(attn_out + s)
        return fused.mean(dim=1), attn_w





