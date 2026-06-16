from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    # Architecture hyperparameters — fill from paper
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1


@dataclass
class TrainConfig:
    # Training hyperparameters — fill from paper
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_epochs: int = 100
    warmup_steps: int = 4000
    grad_clip: float = 1.0
    dataset: str = ""
    data_dir: str = "data"
    output_dir: str = "outputs"
    log_every: int = 100
    save_every: int = 1


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 42
    device: str = "cuda"
