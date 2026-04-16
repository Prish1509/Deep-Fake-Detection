
"""
Temporal Transformer encoder for cross-frame modeling.

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.settings import TEMPORAL_HEADS, TEMPORAL_LAYERS, TEMPORAL_FF_DIM, TRANSFORMER_DROPOUT


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TemporalTransformer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, dropout=TRANSFORMER_DROPOUT)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=TEMPORAL_HEADS,
            dim_feedforward=TEMPORAL_FF_DIM,
            dropout=TRANSFORMER_DROPOUT,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=TEMPORAL_LAYERS)

    def forward(self, x):
        x_enc = self.pos_enc(x)
        out = self.encoder(x_enc)

        with torch.no_grad():
            d_k = x.size(-1) // TEMPORAL_HEADS
            scores = torch.bmm(x_enc, x_enc.transpose(1, 2)) / math.sqrt(d_k)
            attn = F.softmax(scores, dim=-1)

        return out, attn




