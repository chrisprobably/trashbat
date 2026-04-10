"""Shared dataset loading for all models. Preprocessing is model-specific."""

import torch
from pathlib import Path
from PIL import Image
from typing import Callable

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
DATASET_PATH = Path('./dataset-resized')


def load_trashnet(preprocess: Callable[[Image.Image], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Iterate over all dataset images, apply the caller's preprocess function,
    and return (X, Y) tensors.

    preprocess: takes a PIL Image, returns a 1-D float Tensor
    """
    if not DATASET_PATH.exists():
        raise SystemExit(
            f"Dataset not found at '{DATASET_PATH}'. "
            "See README for instructions on how to download and unzip it."
        )
    X_list, Y_list = [], []
    print("Loading dataset...")
    for idx, cls in enumerate(CLASSES):
        folder = DATASET_PATH / cls
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if path.suffix.lower() == '.jpg':
                X_list.append(preprocess(Image.open(path)))
                Y_list.append(idx)
    X = torch.stack(X_list)
    Y = torch.tensor(Y_list)
    print(f"Loaded {len(Y)} images across {len(CLASSES)} classes.")
    return X, Y
