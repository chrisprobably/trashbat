"""
CLI for training models.

Usage:
    python train.py model1              # train a specific model (skips if weights exist)
    python train.py model1 model2       # train multiple models
    python train.py --all               # train all model*.py files
    python train.py --force model1      # retrain even if weights already exist
"""

import argparse
import importlib
import sys
from pathlib import Path

MODELS_DIR = Path("models")
WEIGHTS_DIR = Path("weights")
sys.path.insert(0, str(MODELS_DIR))


def train_model(name: str, force: bool = False) -> None:
    if not (MODELS_DIR / f"{name}.py").exists():
        print(f"Error: models/{name}.py not found.", file=sys.stderr)
        sys.exit(1)
    weights_file = WEIGHTS_DIR / f"{name}.pt"
    if not force and weights_file.exists():
        print(f"\n=== Skipping {name} (weights already exist, use --force to retrain) ===")
        return
    print(f"\n=== Training {name} ===")
    mod = importlib.import_module(name)
    mod.Model().train()


def main():
    parser = argparse.ArgumentParser(description="Train TrashBat models.")
    parser.add_argument("models", nargs="*", help="Model names to train (e.g. model1 model2)")
    parser.add_argument("--all", action="store_true", help="Train all model*.py files")
    parser.add_argument("--force", action="store_true", help="Retrain even if weights already exist")
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
        train_model(name, force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
