import torch


def mean_squared_error(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return ((probs - targets) ** 2).mean()


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, labels)

