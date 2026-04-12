import torch


def mean_squared_error(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return ((probs - targets) ** 2).mean()
