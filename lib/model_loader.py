"""Utility for loading model modules from file paths."""

import importlib.util
import types
from pathlib import Path


def import_model_module(path: Path) -> types.ModuleType:
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
