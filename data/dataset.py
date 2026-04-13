"""Shared dataset loading for all models. Preprocessing is model-specific."""

import torch
from pathlib import Path
from PIL import Image
from typing import Callable

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
DATASET_PATH = Path("./data/dataset-resized")


def load_stratified_data(
    preprocess: Callable[[Image.Image], torch.Tensor],
    train_preprocess: Callable[[Image.Image], torch.Tensor] | None = None,
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

    preprocess: takes a PIL Image, returns a 1-D float Tensor. Applied to
        validation and test splits, and to the training split when
        train_preprocess is not provided.
    train_preprocess: optional separate callable for the training split,
        typically `preprocess` composed with random data augmentation.
    """
    if not DATASET_PATH.exists():
        raise SystemExit(
            f"Dataset not found at '{DATASET_PATH}'. "
            "See README for instructions on how to unzip it."
        )

    train_pp = train_preprocess if train_preprocess is not None else preprocess

    training_X, training_Y = [], []
    validation_X, validation_Y = [], []
    test_X, test_Y = [], []

    print("Loading dataset...")
    for idx, cls in enumerate(CLASSES):
        folder = DATASET_PATH / cls
        if not folder.exists():
            continue
        paths = [p for p in folder.iterdir() if p.suffix.lower() == ".jpg"]
        n = len(paths)
        n_training = round(n * 0.70)
        n_validation = round(n * 0.15)

        train_paths = paths[:n_training]
        val_paths = paths[n_training : n_training + n_validation]
        test_paths = paths[n_training + n_validation :]

        training_X.extend(train_pp(Image.open(p)) for p in train_paths)
        training_Y.extend([idx] * n_training)

        validation_X.extend(preprocess(Image.open(p)) for p in val_paths)
        validation_Y.extend([idx] * n_validation)

        test_X.extend(preprocess(Image.open(p)) for p in test_paths)
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
