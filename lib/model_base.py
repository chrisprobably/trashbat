"""Abstract base class that all trash-classification models must implement."""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from PIL import Image

from data.dataset import CLASSES


class TrashModel(ABC):

    @property
    @abstractmethod
    def weights_path(self) -> Path: ...

    @abstractmethod
    def preprocess(self, img: Image.Image) -> torch.Tensor: ...

    def __init__(self):
        self._weights: torch.Tensor | None = None
        self._bias: torch.Tensor | None = None
        if self.weights_path.exists():
            self._load()

    def _load(self):
        state = torch.load(self.weights_path, weights_only=True)
        self._weights = state["weights"]
        self._bias = state["bias"]

    def _save(self, weights: torch.Tensor, bias: torch.Tensor) -> None:
        self.weights_path.parent.mkdir(exist_ok=True)
        torch.save({"weights": weights, "bias": bias}, self.weights_path)
        print(f"{self.weights_path.stem}: weights saved to {self.weights_path}")
        self._load()

    @property
    def meta_path(self) -> Path:
        return self.weights_path.with_suffix(".meta.json")

    def _save_meta(self, key: str, value: str | float | int) -> None:
        self.weights_path.parent.mkdir(exist_ok=True)
        meta = self._load_meta()
        meta[key] = value
        with self.meta_path.open("w") as f:
            json.dump(meta, f)

    def _load_meta(self) -> dict:
        if not self.meta_path.exists():
            return {}
        with self.meta_path.open() as f:
            return json.load(f)

    @abstractmethod
    def train(self) -> None:
        """Train on the full dataset and save weights to disk."""
        ...

    def predict(self, img: Image.Image) -> dict:
        """
        Run inference on a PIL image.

        Returns:
            {
                'prediction': str,               # winning class name
                'probabilities': {str: float},   # all 6 class scores
            }
        Raises RuntimeError if the model has not been trained yet.
        """
        if self._weights is None or self._bias is None:
            raise RuntimeError(
                f"{self.weights_path.stem} has not been trained. "
                f"Run: python train.py {self.weights_path.stem}"
            )
        x = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(
                torch.mm(x, self._weights) + self._bias, dim=1
            ).squeeze()
        idx = int(torch.argmax(probs).item())
        return {
            "prediction": CLASSES[idx],
            "probabilities": {
                cls: round(probs[i].item(), 4) for i, cls in enumerate(CLASSES)
            },
        }
