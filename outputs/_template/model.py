"""
{Paper Title} — Standalone Model

Self-contained PyTorch implementation of the model architecture.
No local imports — all dependencies are torch + standard library.

Paper: {arxiv URL or citation}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    # Architecture hyperparameters — values from paper
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1


class ComponentName(nn.Module):
    """One major sub-component. Replace with paper's actual component."""

    def __init__(self, config: ModelConfig):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] → [B, T, D]
        return x


class PaperModel(nn.Module):
    """
    Top-level model.
    Input:  [B, T]   token ids (or appropriate input)
    Output: [B, T, vocab_size]  (or appropriate output)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Replace with paper architecture")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    config = ModelConfig()
    model = PaperModel(config)
    print(f"Parameters: {count_parameters(model):,}")

    # Sample forward pass
    x = torch.zeros(2, 16, dtype=torch.long)  # [B=2, T=16]
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
