"""Abstract base class for all paper reproduction models."""
import os
import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Optional


class BaseModel(nn.Module):
    """
    Subclass this for every paper's model.

    Minimal contract:
        - implement forward()
        - call super().__init__() with the config object

    Provides:
        - save / load helpers (checkpoint-compatible)
        - parameter count utility
        - device-aware to() wrapper
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"model_state": self.state_dict(), "config": self.config}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "BaseModel":
        ckpt = torch.load(path, map_location=map_location)
        model = cls(ckpt["config"])
        model.load_state_dict(ckpt["model_state"])
        return model

    # ── Utilities ────────────────────────────────────────────────────────────

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )

    def freeze(self) -> "BaseModel":
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    def unfreeze(self) -> "BaseModel":
        for p in self.parameters():
            p.requires_grad_(True)
        return self

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"params={self.num_parameters():,}, "
            f"trainable={self.num_parameters(trainable_only=True):,})"
        )
