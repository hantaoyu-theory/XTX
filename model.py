# model.py
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

def causal_mask(T: int, device):
    # disallow attending to future positions
    mask = torch.full((T, T), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask  # (T, T)

class LOBTransformer(nn.Module):
    def __init__(self, in_feats: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_ff: int = 256, pdrop: float = 0.1, use_cross_feature_attn: bool = True):
        super().__init__()
        self.in_proj = nn.Linear(in_feats, d_model)
        self.use_cross_feature_attn = use_cross_feature_attn
        
        # Dual-stream architecture: temporal + cross-feature attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=pdrop, batch_first=True, norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers//2)
        
        if use_cross_feature_attn:
            # Cross-feature attention (attend across features at each time step)
            cross_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead//2, dim_feedforward=dim_ff//2,
                dropout=pdrop, batch_first=True, norm_first=True
            )
            self.cross_feature_encoder = nn.TransformerEncoder(cross_layer, num_layers=num_layers//2)
        
        self.posenc = PositionalEncoding(d_model)
        # Multi-head prediction (price movement + volatility + liquidity)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )
        self._causal_masks = {}  # Cache masks by sequence length

    def _get_causal_mask(self, T, device):
        if T not in self._causal_masks:
            mask = torch.full((T, T), float('-inf'), device=device)
            mask = torch.triu(mask, diagonal=1)
            self._causal_masks[T] = mask
        return self._causal_masks[T]

    def forward(self, x):  # x: (B, F, T)
        B, F, T = x.shape
        x = x.transpose(1, 2)           # (B, T, F)
        x = self.in_proj(x)              # (B, T, D)
        x = self.posenc(x)               # (B, T, D)
        
        # Temporal attention (causal)
        mask = self._get_causal_mask(T, x.device)
        h_temporal = self.temporal_encoder(x, mask=mask)  # (B, T, D)
        
        # Cross-feature attention (if enabled)
        if self.use_cross_feature_attn:
            # Reshape to treat each time step's features separately
            h_reshaped = h_temporal.view(B * T, 1, -1)  # (B*T, 1, D)
            # Apply cross-feature attention (no causal mask - features can attend to each other)
            h_cross = self.cross_feature_encoder(h_reshaped)  # (B*T, 1, D)
            h_final = h_cross.view(B, T, -1)  # (B, T, D)
        else:
            h_final = h_temporal
        
        h_last = h_final[:, -1]          # (B, D) -> predict for last tick in window
        return self.head(h_last).squeeze(-1)  # (B,)
