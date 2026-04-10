"""
CLI for training models.

Usage:
    python train.py model1              # train a specific model
    python train.py model1 model2       # train multiple models
    python train.py --all               # train all model*.py files
"""

import argparse
import importlib
import sys
from pathlib import Path

MODELS_DIR = Path("models")
sys.path.insert(0, str(MODELS_DIR))


def train_model(name: str) -> None:
    if not (MODELS_DIR / f"{name}.py").exists():
        print(f"Error: models/{name}.py not found.", file=sys.stderr)
        sys.exit(1)
    print(f"\n=== Training {name} ===")
    mod = importlib.import_module(name)
    mod.Model().train()


def main():
    parser = argparse.ArgumentParser(description="Train TrashBat models.")
    parser.add_argument("models", nargs="*", help="Model names to train (e.g. model1 model2)")
    parser.add_argument("--all", action="store_true", help="Train all model*.py files")
    args = parser.parse_args()

    if args.all:
        names = sorted(p.stem for p in MODELS_DIR.glob("model*.py"))
        if not names:
            print("No model*.py files found.")
            sys.exit(1)
    elif args.models:
        names = args.models
    else:
        parser.print_help()
        sys.exit(1)

    for name in names:
        train_model(name)

    print("\nDone.")


if __name__ == "__main__":
    main()
