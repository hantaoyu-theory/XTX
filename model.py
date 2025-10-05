# model.py
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Learned absolute positional embeddings.

    Replaces fixed sinusoidal encoding with a trainable nn.Embedding(max_len, d_model).
    Forward expects x of shape (B, T, D) and adds position embeddings for [0..T-1].
    """
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        self.max_len = max_len
        self.pe = nn.Embedding(max_len, d_model)
        # Default nn.Embedding init is suitable; optionally could scale
        # nn.init.normal_(self.pe.weight, mean=0.0, std=d_model ** -0.5)

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len {self.max_len} for positional embeddings")
        pos = torch.arange(T, device=x.device)
        pos_emb = self.pe(pos).unsqueeze(0)  # (1, T, D)
        return x + pos_emb

def causal_mask(T: int, device):
    # disallow attending to future positions
    mask = torch.full((T, T), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask  # (T, T)

class LOBTransformer(nn.Module):
    def __init__(self, in_feats: int, d_model: int = 128, nhead: int = 2,
                 num_layers: int = 2, dim_ff: int = 256, pdrop: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_feats, d_model)
        
        # Temporal attention encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=pdrop, batch_first=True, norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
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
        
        h_last = h_temporal[:, -1]       # (B, D) -> predict for last tick in window
        return self.head(h_last).squeeze(-1)  # (B,)
