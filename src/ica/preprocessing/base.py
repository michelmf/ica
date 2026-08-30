"""Preprocessing methods to compute ICA."""
from abc import ABC, abstractmethod

from torch import Tensor
from torch.utils.data import Dataset


class BasePreprocessor(ABC):
    """Base processor class for multiple sources."""
    @abstractmethod
    def transform(self, dataset: Dataset[Tensor]) -> Tensor:
        """Apply preprocessing to the loaded data and return the result."""
