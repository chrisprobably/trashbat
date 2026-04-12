"""
Logistic Regression with MSE Loss, letterboxd images and a large (0.1) learning rate
"""

import torch
from pathlib import Path
from typing import cast
from PIL import Image

from data.dataset import CLASSES, load_trashnet
from lib.model_base import TrashModel
from lib.transforms import resize_med_letterbox


class Model(TrashModel):
    LEARNING_RATE = 0.1
    MAX_ITERATIONS = 10000
    PATIENCE = 1000
    MIN_DELTA = 1e-6
    transform = resize_med_letterbox

    @property
    def weights_path(self) -> Path:
        return Path("weights") / (Path(__file__).stem + ".pt")

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return cast(torch.Tensor, self.transform(img)).view(-1)

    def train(self) -> None:
        X, Y = load_trashnet(self.preprocess)

        # One-hot encode targets for MSE loss
        targets = torch.zeros(len(Y), len(CLASSES))
        targets.scatter_(1, Y.unsqueeze(1), 1.0)

        weights = torch.randn((X.shape[1], len(CLASSES))) * 0.01
        weights.requires_grad_(True)
        bias = torch.zeros(len(CLASSES), requires_grad=True)

        best_loss = float("inf")
        epochs_without_improvement = 0

        print(f"Starting deep training (Max: {self.MAX_ITERATIONS})...")

        for epoch in range(self.MAX_ITERATIONS):
            logits = torch.mm(X, weights) + bias
            probs = torch.softmax(logits, dim=1)
            loss = ((probs - targets) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                if weights.grad is None or bias.grad is None:
                    raise RuntimeError("Gradients missing after backward()")
                w_grad = weights.grad
                b_grad = bias.grad
                weights -= w_grad * self.LEARNING_RATE
                bias -= b_grad * self.LEARNING_RATE
                w_grad.zero_()
                b_grad.zero_()

            # --- EARLY STOPPING LOGIC ---
            current_loss = loss.item()

            # Only consider it an improvement if it drops by more than MIN_DELTA
            if current_loss < (best_loss - self.MIN_DELTA):
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

            if epochs_without_improvement >= self.PATIENCE:
                print(
                    f"\n[Terminated] No significant improvement after {self.PATIENCE} epochs."
                )
                print(f"Final Epoch: {epoch} | Best Loss: {best_loss:.6f}")
                break

        self._save(weights.detach(), bias.detach())
