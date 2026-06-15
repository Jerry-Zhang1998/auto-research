import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from config import ModelConfig


class ExampleModel(nn.Module):
    """Template model — replace with paper architecture."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Define layers here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, ...] → [B, ...]
        raise NotImplementedError("Replace with paper architecture")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from config import ModelConfig
    cfg = ModelConfig()
    model = ExampleModel(cfg)
    print(f"Parameters: {count_parameters(model):,}")
