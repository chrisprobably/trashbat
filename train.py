"""
CLI for training models.

Usage:
    python train.py model1              # train a specific model (skips if weights exist)
    python train.py model1 model2       # train multiple models
    python train.py --all               # train all models in the models/ directory
    python train.py --force model1      # retrain even if weights already exist
"""

import argparse
import sys
from pathlib import Path

from lib.model_loader import import_model_module

MODELS_DIR = Path("models")
WEIGHTS_DIR = Path("weights")


def train_model(name: str, force: bool = False) -> None:
    path = MODELS_DIR / f"{name}.py"
    if not path.exists():
        print(f"Error: models/{name}.py not found.", file=sys.stderr)
        sys.exit(1)
    weights_file = WEIGHTS_DIR / f"{name}.pt"
    if not force and weights_file.exists():
        print(f"\n=== Skipping {name} (weights already exist, use --force to retrain) ===")
        return
    print(f"\n=== Training {name} ===")
    import_model_module(path).Model().train()


def main():
    parser = argparse.ArgumentParser(description="Train TrashBat models.")
    parser.add_argument("models", nargs="*", help="Model names to train (e.g. model1 model2)")
    parser.add_argument("--all", action="store_true", help="Train all models in the models/ directory")
    parser.add_argument("--force", action="store_true", help="Retrain even if weights already exist")
    args = parser.parse_args()

    if args.all:
        names = sorted(
            p.relative_to(MODELS_DIR).with_suffix("").as_posix()
            for p in MODELS_DIR.rglob("*.py")
            if p.name != "__init__.py"
        )
        if not names:
            print("No model files found in models/.")
            sys.exit(1)
    elif args.models:
        names = args.models
    else:
        parser.print_help()
        sys.exit(1)

    for name in names:
        train_model(name, force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
