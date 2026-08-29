"""Base class for ICA methods."""
from abc import ABC, abstractmethod
from typing import Self

from torch import Tensor


class BaseICA(ABC):
    """Base abstract class to implement all ICA methods."""

    @abstractmethod
    def fit(self, X: Tensor) -> Self: ...

    @abstractmethod
    def transform(self, X: Tensor) -> Tensor: ...

    def fit_transform(self, X: Tensor) -> Tensor:
        """Fit and transform in a single command"""
        self.fit(X)
        return self.transform(X)
