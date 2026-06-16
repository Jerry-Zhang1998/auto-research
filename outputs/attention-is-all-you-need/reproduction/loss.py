"""
Label-Smoothing Cross-Entropy Loss (Section 5.4).

The paper uses ε_ls=0.1: distributes 0.1 probability mass uniformly across
all vocabulary tokens, leaving (1 - ε_ls) for the true token.
This improves BLEU/accuracy despite hurting perplexity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from config import TrainConfig


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing.

    Target distribution:
        q(k | x) = (1 - ε_ls) * δ(k = y) + ε_ls / V
    where V = vocab_size, y = true token.

    Padding positions (pad_token_id) are excluded from the loss.
    """

    def __init__(self, config: TrainConfig, vocab_size: int, pad_token_id: int = 0):
        super().__init__()
        self.vocab_size  = vocab_size
        self.eps         = config.label_smoothing     # ε_ls = 0.1
        self.pad_id      = pad_token_id
        self.confidence  = 1.0 - config.label_smoothing

    def forward(
        self,
        logits: torch.Tensor,   # [B, T, V] — decoder output logits
        targets: torch.Tensor,  # [B, T]    — target token IDs
    ) -> Dict[str, torch.Tensor]:
        V = logits.size(-1)

        # Flatten to [B*T, V] and [B*T]
        logits_flat  = logits.contiguous().view(-1, V)             # [N, V]
        targets_flat = targets.contiguous().view(-1)               # [N]

        log_probs = F.log_softmax(logits_flat, dim=-1)             # [N, V]

        # Build smoothed target distribution: ε/V everywhere, (1-ε)+ε/V at true token
        smooth_val = self.eps / V
        smoothed   = torch.full_like(log_probs, smooth_val)        # [N, V]
        smoothed.scatter_(1, targets_flat.unsqueeze(1).clamp(min=0), self.confidence + smooth_val)

        # KL divergence: -Σ q(k) log p(k)
        raw_loss = -(smoothed * log_probs).sum(dim=-1)             # [N]

        # Mask padding positions
        non_pad = targets_flat.ne(self.pad_id)                     # [N]
        loss    = raw_loss.masked_select(non_pad).mean()

        return {
            "total":    loss,
            "ce_loss":  loss,   # alias for logging
        }
