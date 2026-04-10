"""
Model 1: Logistic Regression with Cross-Entropy Loss
Single linear layer (W, b), trained with manual full-batch gradient descent.
No nn.Module, no optimizer.
"""

import torch
import torchvision.transforms as transforms
from pathlib import Path
from typing import cast
from PIL import Image

from dataset import CLASSES, load_trashnet
from model_base import TrashModel

IMG_SIZE = 64
WEIGHTS_PATH = Path('weights/model1.pt')

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])


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
        self._weights = state['weights']
        self._bias = state['bias']

    def train(self) -> None:
        X, Y = load_trashnet(_preprocess)

        weights = torch.randn((X.shape[1], len(CLASSES))) * 0.01
        weights.requires_grad_(True)
        bias = torch.zeros(len(CLASSES), requires_grad=True)

        criterion = torch.nn.CrossEntropyLoss()
        lr = 0.01

        print("Model 1: training (500 epochs, cross-entropy loss)...")
        for epoch in range(500):
            logits = torch.mm(X, weights) + bias
            loss = criterion(logits, Y)
            loss.backward()

            with torch.no_grad():
                if weights.grad is None or bias.grad is None:
                    raise RuntimeError("Gradients missing after backward()")
                w_grad = weights.grad
                b_grad = bias.grad
                weights -= w_grad * lr
                bias -= b_grad * lr
                w_grad.zero_()
                b_grad.zero_()

            if epoch % 100 == 0:
                with torch.no_grad():
                    acc = (torch.argmax(logits, dim=1) == Y).float().mean()
                print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Acc: {acc.item()*100:.1f}%")

        WEIGHTS_PATH.parent.mkdir(exist_ok=True)
        torch.save({'weights': weights.detach(), 'bias': bias.detach()}, WEIGHTS_PATH)
        print(f"Model 1: weights saved to {WEIGHTS_PATH}")
        self._load()

    def predict(self, img: Image.Image) -> dict:
        if self._weights is None or self._bias is None:
            raise RuntimeError("Model 1 has not been trained. Run: python train.py model1")
        x = _preprocess(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(torch.mm(x, self._weights) + self._bias, dim=1).squeeze()
        idx = int(torch.argmax(probs).item())
        return {
            'prediction': CLASSES[idx],
            'probabilities': {cls: round(probs[i].item(), 4) for i, cls in enumerate(CLASSES)},
        }
