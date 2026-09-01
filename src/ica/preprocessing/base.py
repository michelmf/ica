"""Preprocessing methods to compute ICA."""
from abc import ABC, abstractmethod
from typing import Self

from torch import Tensor

class BasePreprocessor(ABC):
    """Base processor class for multiple sources."""

    def fit(self, X: Tensor) -> Self:
        return self

    @abstractmethod
    def transform(self, X: Tensor) -> Tensor:
        """Apply preprocessing to X and return the result."""
