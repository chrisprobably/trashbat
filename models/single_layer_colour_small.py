"""
Logistic Regression with MSE Loss
but uses a 70/15/15 training/validation/test split
"""

import torch
from pathlib import Path
from typing import cast
from PIL import Image

from data.dataset import CLASSES, load_stratified_data
from lib.model_base import TrashModel
from lib.transforms import resize_small_colour


class Model(TrashModel):
    LEARNING_RATE = 0.01
    MAX_ITERATIONS = 10000
    PATIENCE = 500
    MIN_DELTA = 1e-6
    transform = resize_small_colour

    @property
    def weights_path(self) -> Path:
        return Path("weights") / (Path(__file__).stem + ".pt")

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return cast(torch.Tensor, self.transform(img)).view(-1)

    def train(self) -> None:
        (X_training, Y_training), (X_validation, Y_validation), (X_test, Y_test) = (
            load_stratified_data(self.preprocess)
        )

        # One-hot encode targets for MSE loss
        targets_training = torch.zeros(len(Y_training), len(CLASSES))
        targets_training.scatter_(1, Y_training.unsqueeze(1), 1.0)

        weights = torch.randn((X_training.shape[1], len(CLASSES))) * 0.01
        weights.requires_grad_(True)
        bias = torch.zeros(len(CLASSES), requires_grad=True)

        best_validation_loss = float("inf")
        epochs_without_improvement = 0

        print(f"Starting deep training (Max: {self.MAX_ITERATIONS})...")

        for epoch in range(self.MAX_ITERATIONS):
            logits = torch.mm(X_training, weights) + bias
            probs = torch.softmax(logits, dim=1)
            loss = ((probs - targets_training) ** 2).mean()
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

            # --- EARLY STOPPING LOGIC (validation loss) ---
            with torch.no_grad():
                validation_logits = torch.mm(X_validation, weights) + bias
                validation_probs = torch.softmax(validation_logits, dim=1)
                targets_validation = torch.zeros(len(Y_validation), len(CLASSES))
                targets_validation.scatter_(1, Y_validation.unsqueeze(1), 1.0)
                validation_loss = (
                    ((validation_probs - targets_validation) ** 2).mean().item()
                )

            # Only consider it an improvement if it drops by more than MIN_DELTA
            if validation_loss < (best_validation_loss - self.MIN_DELTA):
                best_validation_loss = validation_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Log every 500 epochs to keep the console clean
            if epoch % 500 == 0:
                with torch.no_grad():
                    training_acc = (
                        (torch.argmax(logits, dim=1) == Y_training).float().mean()
                    )
                    validation_acc = (
                        (torch.argmax(validation_logits, dim=1) == Y_validation)
                        .float()
                        .mean()
                    )
                print(
                    f"  Epoch {epoch:5d} | Training Loss: {loss.item():.6f} | Validation Loss: {validation_loss:.6f} | Training Acc: {training_acc.item() * 100:.1f}% | Validation Acc: {validation_acc.item() * 100:.1f}%"
                )

            if epochs_without_improvement >= self.PATIENCE:
                print(
                    f"\n[Terminated] No significant improvement after {self.PATIENCE} epochs."
                )
                print(
                    f"Final Epoch: {epoch} | Best Validation Loss: {best_validation_loss:.6f}"
                )
                break

        with torch.no_grad():
            test_logits = torch.mm(X_test, weights) + bias
            test_acc = (torch.argmax(test_logits, dim=1) == Y_test).float().mean()
        print(f"Test Acc: {test_acc.item() * 100:.1f}%")
        self._save_meta("test_accuracy", f"{test_acc.item() * 100:.1f}%")
        self._save(weights.detach(), bias.detach())
