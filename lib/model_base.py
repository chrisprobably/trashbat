"""Abstract base class that all trash-classification models must implement."""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from data.dataset import CLASSES


class TrashModel(ABC):

    @property
    @abstractmethod
    def weights_path(self) -> Path: ...

    @abstractmethod
    def preprocess(self, img: Image.Image) -> torch.Tensor: ...

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute logits from a batch of pre-processed feature vectors.

        Uses ``self._weights`` / ``self._biases`` which must already be populated
        (either loaded from disk or via ``_save`` during training).
        """
        ...

    def __init__(self):
        self._weights: list[torch.Tensor] = []
        self._biases: list[torch.Tensor] = []
        if self.weights_path.exists():
            self._load()

    def _load(self):
        state = torch.load(self.weights_path, weights_only=True)
        self._weights = list(state["weights"])
        self._biases = list(state["biases"])

    def _save(
        self, weights: list[torch.Tensor], biases: list[torch.Tensor]
    ) -> None:
        self.weights_path.parent.mkdir(exist_ok=True)
        torch.save({"weights": weights, "biases": biases}, self.weights_path)
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

    @property
    def confusion_matrix_path(self) -> Path:
        return self.weights_path.with_suffix(".confusion.png")

    @property
    def loss_plot_path(self) -> Path:
        return self.weights_path.with_suffix(".loss.png")

    def _plot_loss_history(
        self, loss_history: list[tuple[int, float, float]]
    ) -> None:
        if not loss_history:
            return

        epochs_logged, train_losses, val_losses = zip(*loss_history)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs_logged, train_losses, label="Training Loss")
        ax.plot(epochs_logged, val_losses, label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training vs Validation Loss")
        ax.legend()
        self.loss_plot_path.parent.mkdir(exist_ok=True)
        fig.savefig(self.loss_plot_path, bbox_inches="tight")
        plt.close(fig)
        print(
            f"{self.weights_path.stem}: loss plot saved to {self.loss_plot_path}"
        )

    def _plot_confusion_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        with torch.no_grad():
            logits = self.forward(X)
            predictions = torch.argmax(logits, dim=1)

        cm = confusion_matrix(Y.numpy(), predictions.numpy())

        fig = plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASSES,
            yticklabels=CLASSES,
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Waste Classification Confusion Matrix")
        self.confusion_matrix_path.parent.mkdir(exist_ok=True)
        fig.savefig(self.confusion_matrix_path, bbox_inches="tight")
        plt.close(fig)
        print(
            f"{self.weights_path.stem}: confusion matrix saved to {self.confusion_matrix_path}"
        )

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
        if not self._weights or not self._biases:
            raise RuntimeError(
                f"{self.weights_path.stem} has not been trained. "
                f"Run: python train.py {self.weights_path.stem}"
            )
        x = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.forward(x), dim=1).squeeze()
        idx = int(torch.argmax(probs).item())
        return {
            "prediction": CLASSES[idx],
            "probabilities": {
                cls: round(probs[i].item(), 4) for i, cls in enumerate(CLASSES)
            },
        }
