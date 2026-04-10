"""Abstract base class that all trash-classification models must implement."""

from abc import ABC, abstractmethod
from PIL import Image


class TrashModel(ABC):

    @abstractmethod
    def train(self) -> None:
        """Train on the full dataset and save weights to disk."""
        ...

    @abstractmethod
    def predict(self, img: Image.Image) -> dict:
        """
        Run inference on a PIL image.

        Returns:
            {
                'prediction': str,               # winning class name
                'probabilities': {str: float},   # all 6 class scores
            }
        Raises RuntimeError if the model has not been trained yet.
        """
        ...
