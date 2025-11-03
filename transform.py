# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== 可复用的注意力池化（保持与你原模型一致风格） ======
class AttnPool(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):               # x: (B, T, H)
        w = self.fc(x).squeeze(-1)      # (B, T)
        a = torch.softmax(w, dim=1).unsqueeze(-1)
        return (x * a).sum(dim=1)       # (B, H)

# ====== 正弦位置编码 ======
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, H)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # 不参与训练

    def forward(self, x):               # x: (B, T, H)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)  # (1,T,H) + (B,T,H)

# ====== Transformer 多任务模型（对比基线） ======
class TransformerMultiTask(nn.Module):
    """
    输入:  x (B, T, F)
    输出:  logits (B, num_classes), right (B,), left (B,)
    """
    def __init__(self,
                 input_size=44,
                 num_classes=7,
                 d_model=128,
                 nhead=4,
                 num_layers=3,
                 dim_feedforward=256,
                 dropout=0.1,
                 max_len=4096,
                 pooling='attn',           # 'cls' | 'mean' | 'attn'
                 attn_hidden=128):
        super().__init__()
        assert pooling in ['cls', 'mean', 'attn']

        # 1) 特征投影到 d_model
        self.input_proj = nn.Linear(input_size, d_model)

        # 2) （可选）CLS token
        self.pooling = pooling
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        # 3) 位置编码（绝对正弦）
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        # 4) Transformer Encoder（Pre-Norm 更稳定）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,          # 直接用 (B,T,H)
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # 5) 池化到全局表征
        if pooling == 'attn':
            self.pool = AttnPool(d_model, hidden=attn_hidden)

        # 6) 共享层 + 三个任务头（与原版尺寸一致）
        self.fc_shared = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        self.fc_cls   = nn.Linear(64, num_classes)
        self.fc_right = nn.Sequential(nn.Dropout(0.1), nn.Linear(64, 1))
        self.fc_left  = nn.Sequential(nn.Dropout(0.1), nn.Linear(64, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):                # x: (B, T, F)
        B, T, _ = x.shape

        # 1) 线性投影
        h = self.input_proj(x)           # (B, T, H)

        # 2) 位置编码 + 可选 CLS
        if self.pooling == 'cls':
            cls = self.cls_token.expand(B, -1, -1)   # (B,1,H)
            h = torch.cat([cls, h], dim=1)           # (B, T+1, H)
        h = self.pos_enc(h)                          # (B, T(+1), H)

        # 3) Transformer 编码
        # 没有 pad 的情况下，不需要 key_padding_mask / attn_mask
        h = self.encoder(h)                          # (B, T(+1), H)

        # 4) 池化为全局表示
        if self.pooling == 'cls':
            pooled = h[:, 0, :]                      # 取 CLS
        elif self.pooling == 'mean':
            pooled = h.mean(dim=1)                   # 均值池化
        else:
            pooled = self.pool(h)                    # 注意力池化

        # 5) 三头输出
        feat = self.fc_shared(pooled)                # (B, 64)
        logits = self.fc_cls(feat)
        out_r  = self.fc_right(feat).squeeze(-1)
        out_l  = self.fc_left(feat).squeeze(-1)
        return logits, out_r, out_l
