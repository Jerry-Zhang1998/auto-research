import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
from config import TrainConfig


class PaperDataset(Dataset):
    """Template dataset — replace with paper's data loading."""

    def __init__(self, config: TrainConfig, split: str = "train"):
        self.config = config
        self.split = split
        # Load data here

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def get_dataloader(config: TrainConfig, split: str = "train") -> DataLoader:
    dataset = PaperDataset(config, split)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
    )
