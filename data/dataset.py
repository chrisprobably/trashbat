"""Shared dataset loading for all models. Preprocessing is model-specific."""

import torch
from pathlib import Path
from PIL import Image
from typing import Callable

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
DATASET_PATH = Path("./data/dataset-resized")


def load_trashnet(
    preprocess: Callable[[Image.Image], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Iterate over all dataset images, apply the caller's preprocess function,
    and return (X, Y) tensors.

    preprocess: takes a PIL Image, returns a 1-D float Tensor
    """
    if not DATASET_PATH.exists():
        raise SystemExit(
            f"Dataset not found at '{DATASET_PATH}'. "
            "See README for instructions on how to unzip it."
        )
    X_list, Y_list = [], []
    print("Loading dataset...")
    for idx, cls in enumerate(CLASSES):
        folder = DATASET_PATH / cls
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if path.suffix.lower() == ".jpg":
                X_list.append(preprocess(Image.open(path)))
                Y_list.append(idx)
    X = torch.stack(X_list)
    Y = torch.tensor(Y_list)
    print(f"Loaded {len(Y)} images across {len(CLASSES)} classes.")
    return X, Y


def load_stratified_data(
    preprocess: Callable[[Image.Image], torch.Tensor],
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    """
    Iterate over all dataset images, apply the caller's preprocess function,
    and return stratified (X, Y) splits for training, validation, and test sets.

    The dataset is split 70/15/15 (training/validation/test). Each class is split
    independently before combining, ensuring an even spread of each class
    across all three sets.

    preprocess: takes a PIL Image, returns a 1-D float Tensor
    """
    if not DATASET_PATH.exists():
        raise SystemExit(
            f"Dataset not found at '{DATASET_PATH}'. "
            "See README for instructions on how to unzip it."
        )

    training_X, training_Y = [], []
    validation_X, validation_Y = [], []
    test_X, test_Y = [], []

    print("Loading dataset...")
    for idx, cls in enumerate(CLASSES):
        folder = DATASET_PATH / cls
        if not folder.exists():
            continue
        tensors = [
            preprocess(Image.open(path))
            for path in folder.iterdir()
            if path.suffix.lower() == ".jpg"
        ]
        n = len(tensors)
        n_training = round(n * 0.70)
        n_validation = round(n * 0.15)

        training_X.extend(tensors[:n_training])
        training_Y.extend([idx] * n_training)

        validation_X.extend(tensors[n_training : n_training + n_validation])
        validation_Y.extend([idx] * n_validation)

        test_X.extend(tensors[n_training + n_validation :])
        test_Y.extend([idx] * (n - n_training - n_validation))

    total = len(training_Y) + len(validation_Y) + len(test_Y)
    print(
        f"Loaded {total} images across {len(CLASSES)} classes "
        f"({len(training_Y)} training / {len(validation_Y)} validation / {len(test_Y)} test)."
    )
    return (
        (torch.stack(training_X), torch.tensor(training_Y)),
        (torch.stack(validation_X), torch.tensor(validation_Y)),
        (torch.stack(test_X), torch.tensor(test_Y)),
    )
