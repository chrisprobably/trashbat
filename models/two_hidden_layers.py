import torch
from pathlib import Path
from typing import cast
from PIL import Image

from data.dataset import CLASSES, load_stratified_data
from lib.model_base import TrashModel
from lib.criteria import mean_squared_error
from lib.transforms import MEDIUM_IMG_SIZE, resize_med_colour


class Model(TrashModel):
    LEARNING_RATE = 0.01
    MAX_ITERATIONS = 10000
    PATIENCE = MAX_ITERATIONS
    MIN_DELTA = 1e-6
    transform = resize_med_colour
    criterion = staticmethod(mean_squared_error)
    INPUT_SIZE = MEDIUM_IMG_SIZE * MEDIUM_IMG_SIZE
    HIDDEN_LAYER_1_SIZE = INPUT_SIZE // 16
    HIDDEN_LAYER_2_SIZE = HIDDEN_LAYER_1_SIZE // 2

    @property
    def weights_path(self) -> Path:
        return Path("weights") / (Path(__file__).stem + ".pt")

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return cast(torch.Tensor, self.transform(img)).view(-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        hidden1 = torch.relu(torch.mm(X, self._weights[0]) + self._biases[0])
        hidden2 = torch.relu(torch.mm(hidden1, self._weights[1]) + self._biases[1])
        return torch.mm(hidden2, self._weights[2]) + self._biases[2]

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
        loss_history: list[tuple[int, float, float]] = []

        print(
            f"Starting deep training (Max: {self.MAX_ITERATIONS}, Hidden layer 1: {self.HIDDEN_LAYER_1_SIZE} Layer 2: {self.HIDDEN_LAYER_2_SIZE})..."
        )

        for epoch in range(self.MAX_ITERATIONS):
            hidden1 = torch.relu(torch.mm(X_training, weights1) + bias1)
            hidden2 = torch.relu(torch.mm(hidden1, weights2) + bias2)
            logits = torch.mm(hidden2, weights3) + bias3
            probs = torch.softmax(logits, dim=1)
            loss = type(self).criterion(probs, targets_training)
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
                    type(self).criterion(validation_probs, targets_validation).item()
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
                loss_history.append((epoch, loss.item(), validation_loss))
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
        self._save_meta("test_accuracy", f"{test_acc.item() * 100:.1f}%")

        self._save(
            [weights1.detach(), weights2.detach(), weights3.detach()],
            [bias1.detach(), bias2.detach(), bias3.detach()],
        )
        self._plot_confusion_matrix(X_validation, Y_validation)
        self._plot_loss_history(loss_history)
