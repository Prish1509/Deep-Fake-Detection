import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

FACE_SIZE = 224
NUM_FRAMES = 16
FEATURE_DIM = 1280
SPATIAL_ATT_R = 16
T_HEADS = 4
T_LAYERS = 2
T_FF = 512
FUSION_DIM = 256
DROPOUT = 0.3
T_DROPOUT = 0.1


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=SPATIAL_ATT_R):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(ch, ch//r, bias=False), nn.ReLU(True),
                                nn.Linear(ch//r, ch, bias=False))
        self.sig = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        a = self.fc(self.ap(x).view(b,c)) + self.fc(self.mp(x).view(b,c))
        return x * self.sig(a).view(b,c,1,1)

class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k//2, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        att = self.sig(self.conv(torch.cat([avg, mx], 1)))
        return x * att, att

class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.ca(x)
        x, sm = self.sa(x)
        return x, sm

class PosEnc(nn.Module):
    def __init__(self, d, maxlen=100, drop=0.1):
        super().__init__()
        self.drop = nn.Dropout(drop)
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(0, maxlen, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])

class TemporalTransformer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.pe = PosEnc(d, dropout=T_DROPOUT)
        layer = nn.TransformerEncoderLayer(d, T_HEADS, T_FF, T_DROPOUT,
                                           batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, T_LAYERS)
    def forward(self, x):
        xp = self.pe(x)
        out = self.enc(xp)
        with torch.no_grad():
            dk = x.size(-1) // T_HEADS
            s = torch.bmm(xp, xp.transpose(1,2)) / math.sqrt(dk)
            attn = F.softmax(s, dim=-1)
        return out, attn

class CrossFusion(nn.Module):
    def __init__(self, sd, td, fd=FUSION_DIM):
        super().__init__()
        self.sp = nn.Linear(sd, fd)
        self.tp = nn.Linear(td, fd)
        self.ca = nn.MultiheadAttention(fd, 4, 0.1, batch_first=True)
        self.ln = nn.LayerNorm(fd)
        self.out_dim = fd
    def forward(self, sf, tf):
        s, t = self.sp(sf), self.tp(tf)
        ao, aw = self.ca(s, t, t)
        return self.ln(ao + s).mean(1), aw

class DualForensics(nn.Module):
    def __init__(self):
        super().__init__()
        D = FEATURE_DIM
        base = models.efficientnet_b0(weights=None)
        self.backbone_feat = base.features
        self.backbone_pool = nn.AdaptiveAvgPool2d(1)
        self.cbam = CBAM(D)
        self.s_pool = nn.AdaptiveAvgPool2d(1)
        self.temporal = TemporalTransformer(D)
        self.fusion = CrossFusion(D, D)
        self.head = nn.Sequential(nn.Linear(self.fusion.out_dim, 256), nn.ReLU(True),
                                  nn.Dropout(DROPOUT), nn.Linear(256, 1))
    def forward(self, x):
        B, N, C, H, W = x.shape
        flat = x.view(B*N, C, H, W)
        fmaps = self.backbone_feat(flat)
        fvecs = self.backbone_pool(fmaps).flatten(1)
        D = fvecs.size(1)
        h, w = fmaps.shape[2], fmaps.shape[3]
        att_maps, s_maps = self.cbam(fmaps)
        sf = self.s_pool(att_maps).flatten(1).view(B, N, D)
        tf_in = fvecs.view(B, N, D)
        tf_out, t_attn = self.temporal(tf_in)
        fused, c_attn = self.fusion(sf, tf_out)
        logits = self.head(fused)
        return logits, {"spatial_maps": s_maps.view(B,N,1,h,w).detach(),
                        "temporal_attn": t_attn.detach(), "cross_attn": c_attn.detach()}
