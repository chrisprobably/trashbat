"""
Logistic Regression with MSE Loss and 2 hidden layers (ReLU activation)
Uses a 70/15/15 training/validation/test split
"""

import torch
from pathlib import Path
from typing import cast
from PIL import Image

from data.dataset import CLASSES, load_stratified_data
from lib.model_base import TrashModel
from lib.transforms import MEDIUM_IMG_SIZE, resize_med_colour
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class Model(TrashModel):
    LEARNING_RATE = 0.01
    MAX_ITERATIONS = 20000
    PATIENCE = 500
    MIN_DELTA = 1e-6
    transform = resize_med_colour
    INPUT_SIZE = MEDIUM_IMG_SIZE * MEDIUM_IMG_SIZE
    HIDDEN_LAYER_1_SIZE = INPUT_SIZE // 16
    HIDDEN_LAYER_2_SIZE = HIDDEN_LAYER_1_SIZE // 2

    def __init__(self):
        self._weights1: torch.Tensor | None = None
        self._bias1: torch.Tensor | None = None
        self._weights2: torch.Tensor | None = None
        self._bias2: torch.Tensor | None = None
        super().__init__()

    @property
    def weights_path(self) -> Path:
        return Path("weights") / (Path(__file__).stem + ".pt")

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return cast(torch.Tensor, self.transform(img)).view(-1)

    def _load(self):
        state = torch.load(self.weights_path, weights_only=True)
        self._weights = state["weights3"]
        self._bias = state["bias3"]
        self._weights1 = state["weights1"]
        self._bias1 = state["bias1"]
        self._weights2 = state["weights2"]
        self._bias2 = state["bias2"]

    def predict(self, img: Image.Image) -> dict:
        if (
            self._weights is None
            or self._bias is None
            or self._weights1 is None
            or self._bias1 is None
            or self._weights2 is None
            or self._bias2 is None
        ):
            raise RuntimeError(
                f"{self.weights_path.stem} has not been trained. "
                f"Run: python train.py {self.weights_path.stem}"
            )
        x = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            hidden1 = torch.relu(torch.mm(x, self._weights1) + self._bias1)
            hidden2 = torch.relu(torch.mm(hidden1, self._weights2) + self._bias2)
            probs = torch.softmax(
                torch.mm(hidden2, self._weights) + self._bias, dim=1
            ).squeeze()
        idx = int(torch.argmax(probs).item())
        return {
            "prediction": CLASSES[idx],
            "probabilities": {
                cls: round(probs[i].item(), 4) for i, cls in enumerate(CLASSES)
            },
        }

    def train(self) -> None:
        (X_training, Y_training), (X_validation, Y_validation), (X_test, Y_test) = (
            load_stratified_data(self.preprocess)
        )

        input_size = X_training.shape[1]

        # One-hot encode targets for MSE loss
        targets_training = torch.zeros(len(Y_training), len(CLASSES))
        targets_training.scatter_(1, Y_training.unsqueeze(1), 1.0)

        # Layer 1: input -> hidden1
        weights1 = torch.randn((input_size, self.HIDDEN_LAYER_1_SIZE)) * 0.01
        weights1.requires_grad_(True)
        bias1 = torch.zeros(self.HIDDEN_LAYER_1_SIZE, requires_grad=True)

        # Layer 2: hidden1 -> hidden2
        weights2 = (
            torch.randn((self.HIDDEN_LAYER_1_SIZE, self.HIDDEN_LAYER_2_SIZE)) * 0.01
        )
        weights2.requires_grad_(True)
        bias2 = torch.zeros(self.HIDDEN_LAYER_2_SIZE, requires_grad=True)

        # Layer 3: hidden2 -> output
        weights3 = torch.randn((self.HIDDEN_LAYER_2_SIZE, len(CLASSES))) * 0.01
        weights3.requires_grad_(True)
        bias3 = torch.zeros(len(CLASSES), requires_grad=True)

        best_validation_loss = float("inf")
        epochs_without_improvement = 0

        print(
            f"Starting deep training (Max: {self.MAX_ITERATIONS}, Hidden layer 1: {self.HIDDEN_LAYER_1_SIZE} Layer 2: {self.HIDDEN_LAYER_2_SIZE})..."
        )

        for epoch in range(self.MAX_ITERATIONS):
            hidden1 = torch.relu(torch.mm(X_training, weights1) + bias1)
            hidden2 = torch.relu(torch.mm(hidden1, weights2) + bias2)
            logits = torch.mm(hidden2, weights3) + bias3
            probs = torch.softmax(logits, dim=1)
            loss = ((probs - targets_training) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                if (
                    weights1.grad is None
                    or bias1.grad is None
                    or weights2.grad is None
                    or bias2.grad is None
                    or weights3.grad is None
                    or bias3.grad is None
                ):
                    raise RuntimeError("Gradients missing after backward()")
                w1_grad = weights1.grad
                b1_grad = bias1.grad
                w2_grad = weights2.grad
                b2_grad = bias2.grad
                w3_grad = weights3.grad
                b3_grad = bias3.grad
                weights1 -= w1_grad * self.LEARNING_RATE
                w1_grad.zero_()
                bias1 -= b1_grad * self.LEARNING_RATE
                b1_grad.zero_()
                weights2 -= w2_grad * self.LEARNING_RATE
                w2_grad.zero_()
                bias2 -= b2_grad * self.LEARNING_RATE
                b2_grad.zero_()
                weights3 -= w3_grad * self.LEARNING_RATE
                w3_grad.zero_()
                bias3 -= b3_grad * self.LEARNING_RATE
                b3_grad.zero_()

            # --- EARLY STOPPING LOGIC (validation loss) ---
            with torch.no_grad():
                validation_hidden1 = torch.relu(
                    torch.mm(X_validation, weights1) + bias1
                )
                validation_hidden2 = torch.relu(
                    torch.mm(validation_hidden1, weights2) + bias2
                )
                validation_logits = torch.mm(validation_hidden2, weights3) + bias3
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
            test_hidden1 = torch.relu(torch.mm(X_test, weights1) + bias1)
            test_hidden2 = torch.relu(torch.mm(test_hidden1, weights2) + bias2)
            test_logits = torch.mm(test_hidden2, weights3) + bias3
            test_acc = (torch.argmax(test_logits, dim=1) == Y_test).float().mean()
        print(f"Test Acc: {test_acc.item() * 100:.1f}%")

        # --- GENERATE CONFUSION MATRIX ---
        with torch.no_grad():
            h1 = torch.relu(torch.mm(X_validation, weights1) + bias1)
            h2 = torch.relu(torch.mm(h1, weights2) + bias2)
            logits = torch.mm(h2, weights3) + bias3
            predictions = torch.argmax(logits, dim=1)

        y_true = Y_validation.numpy()
        y_pred = predictions.numpy()

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 7))
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
        plt.show()

        self.weights_path.parent.mkdir(exist_ok=True)
        torch.save(
            {
                "weights1": weights1.detach(),
                "bias1": bias1.detach(),
                "weights2": weights2.detach(),
                "bias2": bias2.detach(),
                "weights3": weights3.detach(),
                "bias3": bias3.detach(),
            },
            self.weights_path,
        )
        print(f"{self.weights_path.stem}: weights saved to {self.weights_path}")
        self._load()
