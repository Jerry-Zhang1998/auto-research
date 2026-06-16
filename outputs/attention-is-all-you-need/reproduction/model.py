"""
Transformer — "Attention Is All You Need" (Vaswani et al., 2017)
https://arxiv.org/abs/1706.03762

Implements the base model (d_model=512, h=8, N=6) faithfully to the paper:
  - Post-LayerNorm:  x = LayerNorm(x + Dropout(Sublayer(x)))          (Section 3.1)
  - Scaled dot-product attention:  softmax(QKᵀ / √d_k) V              (Eq. 1)
  - Multi-head attention: h=8 heads, d_k=d_v=64                       (Section 3.2)
  - Sinusoidal positional encoding                                     (Section 3.5)
  - Shared weight matrix: encoder embed = decoder embed = output proj  (Section 3.4)
  - Embedding outputs scaled by √d_model                              (Section 3.4)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import ModelConfig


# ── Positional Encoding ────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """PE(pos, 2i) = sin(pos / 10000^{2i/d_model}),  PE(pos, 2i+1) = cos(...)"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                   # [max_len, d_model]
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)   # [max_len, 1]
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )                                                     # [d_model/2]
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))          # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        x = x + self.pe[:, : x.size(1)]                     # [B, T, d_model]
        return self.dropout(x)


# ── Scaled Dot-Product Attention ───────────────────────────────────────────────

class ScaledDotProductAttention(nn.Module):
    """Attention(Q, K, V) = softmax(QKᵀ / √d_k) V   (Eq. 1)"""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        q: torch.Tensor,                    # [B, h, Tq, d_k]
        k: torch.Tensor,                    # [B, h, Tk, d_k]
        v: torch.Tensor,                    # [B, h, Tk, d_v]
        mask: Optional[torch.Tensor] = None,  # [B, h, Tq, Tk] or broadcastable; True = mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [B, h, Tq, Tk]
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)                     # [B, h, Tq, Tk]
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)                          # [B, h, Tq, d_v]
        return out, attn


# ── Multi-Head Attention ───────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    W_i^Q ∈ R^{512×64}, W_i^K ∈ R^{512×64}, W_i^V ∈ R^{512×64}, W^O ∈ R^{512×512}
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads       # 64 for base model
        self.d_v = d_model // n_heads       # 64 for base model

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # W^O ∈ R^{512×512}

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,                     # [B, Tq, d_model]
        key: torch.Tensor,                       # [B, Tk, d_model]
        value: torch.Tensor,                     # [B, Tk, d_model]
        mask: Optional[torch.Tensor] = None,     # [B, 1, Tq, Tk] or [B, 1, 1, Tk]
    ) -> torch.Tensor:                           # [B, Tq, d_model]
        B, Tq = query.size(0), query.size(1)

        # Project and split into h heads: [B, h, T, d_k]
        q = self.W_q(query).view(B, Tq, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(B, key.size(1), self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(B, value.size(1), self.n_heads, self.d_v).transpose(1, 2)

        out, _ = self.attention(q, k, v, mask=mask)   # [B, h, Tq, d_v]

        # Concat heads and project: [B, Tq, d_model]
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.n_heads * self.d_v)
        return self.W_o(out)


# ── Position-wise Feed-Forward Network ────────────────────────────────────────

class PositionWiseFFN(nn.Module):
    """FFN(x) = max(0, x W_1 + b_1) W_2 + b_2   (Section 3.3, Eq. 2)"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        return self.fc2(self.dropout(F.relu(self.fc1(x))))   # [B, T, d_model]


# ── Sub-Layer Connection ───────────────────────────────────────────────────────

class SubLayerConnection(nn.Module):
    """Post-norm residual: x = LayerNorm(x + Dropout(Sublayer(x)))  (Section 3.1)"""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        return self.norm(x + self.dropout(sublayer(x)))


# ── Encoder Layer ─────────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """
    Two sub-layers (Section 3.1):
      1. Multi-head self-attention
      2. Position-wise FFN
    Each with residual + post-LayerNorm.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn       = PositionWiseFFN(d_model, d_ff, dropout=dropout)
        self.sublayer  = nn.ModuleList([
            SubLayerConnection(d_model, dropout),
            SubLayerConnection(d_model, dropout),
        ])

    def forward(
        self,
        x: torch.Tensor,                          # [B, T, d_model]
        src_mask: Optional[torch.Tensor] = None,  # [B, 1, 1, T]
    ) -> torch.Tensor:                            # [B, T, d_model]
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, src_mask))
        x = self.sublayer[1](x, self.ffn)
        return x


# ── Decoder Layer ─────────────────────────────────────────────────────────────

class DecoderLayer(nn.Module):
    """
    Three sub-layers (Section 3.1):
      1. Masked multi-head self-attention (causal)
      2. Multi-head cross-attention (Q from decoder, K/V from encoder)
      3. Position-wise FFN
    Each with residual + post-LayerNorm.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn        = PositionWiseFFN(d_model, d_ff, dropout=dropout)
        self.sublayer   = nn.ModuleList([
            SubLayerConnection(d_model, dropout),
            SubLayerConnection(d_model, dropout),
            SubLayerConnection(d_model, dropout),
        ])

    def forward(
        self,
        x: torch.Tensor,                            # [B, T', d_model]
        enc_out: torch.Tensor,                      # [B, T, d_model]
        tgt_mask: Optional[torch.Tensor] = None,    # [B, 1, T', T']  causal + padding
        src_mask: Optional[torch.Tensor] = None,    # [B, 1, 1, T]    source padding
    ) -> torch.Tensor:                              # [B, T', d_model]
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.cross_attn(x, enc_out, enc_out, src_mask))
        x = self.sublayer[2](x, self.ffn)
        return x


# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_encoder_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,                         # [B, T, d_model]
        mask: Optional[torch.Tensor] = None,     # [B, 1, 1, T]
    ) -> torch.Tensor:                           # [B, T, d_model]
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ── Decoder ───────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_decoder_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,                             # [B, T', d_model]
        enc_out: torch.Tensor,                       # [B, T, d_model]
        tgt_mask: Optional[torch.Tensor] = None,    # [B, 1, T', T']
        src_mask: Optional[torch.Tensor] = None,    # [B, 1, 1, T]
    ) -> torch.Tensor:                              # [B, T', d_model]
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.norm(x)


# ── Mask Utilities ────────────────────────────────────────────────────────────

def make_pad_mask(seq: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """Returns [B, 1, 1, T] bool mask: True = pad position (should be masked)."""
    return (seq == pad_id).unsqueeze(1).unsqueeze(2)   # [B, 1, 1, T]


def make_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """Returns [1, 1, T, T] upper-triangular bool mask: True = future (masked)."""
    mask = torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)              # [1, 1, T, T]


# ── Transformer ───────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    """
    Full encoder-decoder Transformer (Section 3).

    Forward I/O:
        src:  [B, S]  source token IDs
        tgt:  [B, T]  target token IDs (teacher-forced during training)
        →     [B, T, vocab_size]  logits
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config   = config
        self.d_model  = config.d_model
        self.pad_id   = config.pad_token_id

        # Embeddings — shared across encoder input, decoder input, output projection
        vocab_size = max(config.src_vocab_size, config.tgt_vocab_size)
        self.src_embed = nn.Embedding(vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.tgt_embed = nn.Embedding(vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.output_proj = nn.Linear(config.d_model, vocab_size, bias=False)

        if config.share_embeddings:
            # Weight tying: encoder embed = decoder embed = output projection transpose
            self.tgt_embed.weight  = self.src_embed.weight
            self.output_proj.weight = self.src_embed.weight  # Linear weight: [out, in]

        self.src_pe = SinusoidalPositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        self.tgt_pe = SinusoidalPositionalEncoding(config.d_model, config.max_seq_len, config.dropout)

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        src: torch.Tensor,                       # [B, S]
        src_mask: Optional[torch.Tensor] = None,  # [B, 1, 1, S]
    ) -> torch.Tensor:                           # [B, S, d_model]
        # Scale embedding by √d_model before adding PE (Section 3.4)
        x = self.src_embed(src) * math.sqrt(self.d_model)   # [B, S, d_model]
        x = self.src_pe(x)                                   # [B, S, d_model]
        return self.encoder(x, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,                         # [B, T]
        enc_out: torch.Tensor,                     # [B, S, d_model]
        tgt_mask: Optional[torch.Tensor] = None,  # [B, 1, T, T]
        src_mask: Optional[torch.Tensor] = None,  # [B, 1, 1, S]
    ) -> torch.Tensor:                            # [B, T, d_model]
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)   # [B, T, d_model]
        x = self.tgt_pe(x)                                   # [B, T, d_model]
        return self.decoder(x, enc_out, tgt_mask, src_mask)

    def forward(
        self,
        src: torch.Tensor,                       # [B, S] source token IDs
        tgt: torch.Tensor,                       # [B, T] target token IDs
    ) -> torch.Tensor:                           # [B, T, vocab_size] logits
        src_mask = make_pad_mask(src, self.pad_id)                           # [B, 1, 1, S]
        tgt_pad  = make_pad_mask(tgt, self.pad_id)                           # [B, 1, 1, T]
        tgt_caus = make_causal_mask(tgt.size(1), tgt.device)                 # [1, 1, T, T]
        tgt_mask = tgt_pad | tgt_caus                                        # [B, 1, T, T]

        enc_out = self.encode(src, src_mask)                 # [B, S, d_model]
        dec_out = self.decode(tgt, enc_out, tgt_mask, src_mask)  # [B, T, d_model]
        return self.output_proj(dec_out)                     # [B, T, vocab_size]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"Transformer(d_model={self.d_model}, "
            f"n_enc={self.config.n_encoder_layers}, "
            f"n_dec={self.config.n_decoder_layers}, "
            f"n_heads={self.config.n_heads}, "
            f"d_ff={self.config.d_ff}, "
            f"params={self.count_parameters():,})"
        )
