"""Reproducibility utilities."""
import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed:          integer seed value
        deterministic: if True, set CUDNN to deterministic mode
                       (slower, but bit-exact reproducibility across runs)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass   # older PyTorch
