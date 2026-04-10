"""
Logistic Regression with MSE Loss and letterboxed images
"""

import torch
import torchvision.transforms as transforms
from pathlib import Path
from typing import cast
from PIL import Image

from dataset import CLASSES, load_trashnet
from model_base import TrashModel

IMG_SIZE = 64
LEARNING_RATE = 0.01
MAX_ITERATIONS = 10000
PATIENCE = 500
MIN_DELTA = 1e-6
MODEL_NAME = Path(__file__).stem
WEIGHTS_PATH = Path("weights") / (MODEL_NAME + ".pt")

_transform = transforms.Compose(
    [
        # 1. Resize the long side to 64 while maintaining aspect ratio
        # This turns 512x384 into 64x48
        transforms.Resize(IMG_SIZE),
        # 2. Pad the shorter side (48) with black pixels to reach 64x64
        # 'CenterCrop' on a smaller image with 'pad_if_needed' does exactly this!
        transforms.CenterCrop(IMG_SIZE),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)


def _preprocess(img: Image.Image) -> torch.Tensor:
    return cast(torch.Tensor, _transform(img)).view(-1)


class Model(TrashModel):
    def __init__(self):
        self._weights: torch.Tensor | None = None
        self._bias: torch.Tensor | None = None
        if WEIGHTS_PATH.exists():
            self._load()

    def _load(self):
        state = torch.load(WEIGHTS_PATH, weights_only=True)
        self._weights = state["weights"]
        self._bias = state["bias"]

    def train(self) -> None:
        X, Y = load_trashnet(_preprocess)

        # One-hot encode targets for MSE loss
        targets = torch.zeros(len(Y), len(CLASSES))
        targets.scatter_(1, Y.unsqueeze(1), 1.0)

        weights = torch.randn((X.shape[1], len(CLASSES))) * 0.01
        weights.requires_grad_(True)
        bias = torch.zeros(len(CLASSES), requires_grad=True)

        best_loss = float("inf")
        epochs_without_improvement = 0

        print(f"Starting deep training (Max: {MAX_ITERATIONS})...")

        for epoch in range(MAX_ITERATIONS):
            logits = torch.mm(X, weights) + bias
            probs = torch.softmax(logits, dim=1)
            loss = ((probs - targets) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                if weights.grad is None or bias.grad is None:
                    raise RuntimeError("Gradients missing after backward()")
                w_grad = weights.grad
                b_grad = bias.grad
                weights -= w_grad * LEARNING_RATE
                bias -= b_grad * LEARNING_RATE
                w_grad.zero_()
                b_grad.zero_()

            # --- EARLY STOPPING LOGIC ---
            current_loss = loss.item()

            # Only consider it an improvement if it drops by more than MIN_DELTA
            if current_loss < (best_loss - MIN_DELTA):
                best_loss = current_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Log every 500 epochs to keep the console clean
            if epoch % 500 == 0:
                with torch.no_grad():
                    acc = (torch.argmax(logits, dim=1) == Y).float().mean()
                print(
                    f"  Epoch {epoch:5d} | Loss: {current_loss:.6f} | Acc: {acc.item() * 100:.1f}%"
                )

            if epochs_without_improvement >= PATIENCE:
                print(
                    f"\n[Terminated] No significant improvement after {PATIENCE} epochs."
                )
                print(f"Final Epoch: {epoch} | Best Loss: {best_loss:.6f}")
                break

        WEIGHTS_PATH.parent.mkdir(exist_ok=True)
        torch.save({"weights": weights.detach(), "bias": bias.detach()}, WEIGHTS_PATH)
        print(f"{MODEL_NAME}: weights saved to {WEIGHTS_PATH}")
        self._load()

    def predict(self, img: Image.Image) -> dict:
        if self._weights is None or self._bias is None:
            raise RuntimeError(
                f"{MODEL_NAME} has not been trained. Run: python train.py {MODEL_NAME}"
            )
        x = _preprocess(img).unsqueeze(0)
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
