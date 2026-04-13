import math
import torch
from pathlib import Path
from typing import cast
from PIL import Image

from data.dataset import CLASSES, load_stratified_data
from lib.model_base import TrashModel
from lib.criteria import cross_entropy
from lib.transforms import resize_med_colour, MEDIUM_IMG_SIZE


PATCH_SIZE = 8
NUM_PATCHES_PER_SIDE = MEDIUM_IMG_SIZE // PATCH_SIZE
NUM_PATCHES = NUM_PATCHES_PER_SIDE * NUM_PATCHES_PER_SIDE
PATCH_DIM = 3 * PATCH_SIZE * PATCH_SIZE
EMBED_DIM = 128


class Model(TrashModel):
    LEARNING_RATE = 0.01
    MAX_ITERATIONS = 20000
    PATIENCE = 500
    MIN_DELTA = 1e-6
    HIDDEN_SIZE = 64
    transform = resize_med_colour
    criterion = staticmethod(cross_entropy)

    @property
    def weights_path(self) -> Path:
        return Path("weights") / (Path(__file__).stem + ".pt")

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        t = cast(torch.Tensor, self.transform(img))
        # (3, H, W) -> (3, n_side, n_side, P, P) -> (num_patches, 3*P*P)
        patches = t.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        return patches.view(NUM_PATCHES, PATCH_DIM)

    @staticmethod
    def _attend(
        X: torch.Tensor,
        Wq: torch.Tensor,
        Wk: torch.Tensor,
        Wv: torch.Tensor,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
    ) -> torch.Tensor:
        Q = torch.matmul(X, Wq) + bq
        K = torch.matmul(X, Wk) + bk
        V = torch.matmul(X, Wv) + bv
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(EMBED_DIM)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Wq, Wk, Wv, W1, W2 = self._weights
        bq, bk, bv, b1, b2 = self._biases
        attended = self._attend(X, Wq, Wk, Wv, bq, bk, bv)
        flat = attended.reshape(attended.shape[0], -1)
        hidden = torch.relu(torch.mm(flat, W1) + b1)
        return torch.mm(hidden, W2) + b2

    def train(self) -> None:
        (X_training, Y_training), (X_validation, Y_validation), (X_test, Y_test) = (
            load_stratified_data(self.preprocess)
        )

        flat_size = NUM_PATCHES * EMBED_DIM

        # Attention projection matrices (patch_dim -> embed_dim)
        Wq = (torch.randn((PATCH_DIM, EMBED_DIM)) * 0.01).requires_grad_(True)
        Wk = (torch.randn((PATCH_DIM, EMBED_DIM)) * 0.01).requires_grad_(True)
        Wv = (torch.randn((PATCH_DIM, EMBED_DIM)) * 0.01).requires_grad_(True)
        bq = torch.zeros(EMBED_DIM, requires_grad=True)
        bk = torch.zeros(EMBED_DIM, requires_grad=True)
        bv = torch.zeros(EMBED_DIM, requires_grad=True)

        # MLP layers on flattened attention output
        W1 = (torch.randn((flat_size, self.HIDDEN_SIZE)) * 0.01).requires_grad_(True)
        b1 = torch.zeros(self.HIDDEN_SIZE, requires_grad=True)
        W2 = (torch.randn((self.HIDDEN_SIZE, len(CLASSES))) * 0.01).requires_grad_(True)
        b2 = torch.zeros(len(CLASSES), requires_grad=True)

        params = [Wq, Wk, Wv, bq, bk, bv, W1, b1, W2, b2]

        def run(X: torch.Tensor) -> torch.Tensor:
            attended = self._attend(X, Wq, Wk, Wv, bq, bk, bv)
            flat = attended.reshape(attended.shape[0], -1)
            hidden = torch.relu(torch.mm(flat, W1) + b1)
            return torch.mm(hidden, W2) + b2

        best_validation_loss = float("inf")
        epochs_without_improvement = 0

        print(
            f"Starting transformer training (Max: {self.MAX_ITERATIONS}, "
            f"Patches: {NUM_PATCHES}, Patch size: {PATCH_SIZE}, Num patches/side: {NUM_PATCHES_PER_SIDE}, Embed: {EMBED_DIM}, Hidden: {self.HIDDEN_SIZE})..."
        )

        for epoch in range(self.MAX_ITERATIONS):
            logits = run(X_training)
            loss = type(self).criterion(logits, Y_training)
            loss.backward()

            with torch.no_grad():
                for p in params:
                    if p.grad is None:
                        raise RuntimeError("Gradients missing after backward()")
                    grad = p.grad
                    p -= grad * self.LEARNING_RATE
                    grad.zero_()

            with torch.no_grad():
                validation_logits = run(X_validation)
                validation_loss = (
                    type(self).criterion(validation_logits, Y_validation).item()
                )

            if validation_loss < (best_validation_loss - self.MIN_DELTA):
                best_validation_loss = validation_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

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
            test_logits = run(X_test)
            test_acc = (torch.argmax(test_logits, dim=1) == Y_test).float().mean()
        print(f"Test Acc: {test_acc.item() * 100:.1f}%")
        self._save_meta("test_accuracy", f"{test_acc.item() * 100:.1f}%")

        self._save(
            [Wq.detach(), Wk.detach(), Wv.detach(), W1.detach(), W2.detach()],
            [bq.detach(), bk.detach(), bv.detach(), b1.detach(), b2.detach()],
        )
        self._plot_confusion_matrix(X_validation, Y_validation)
