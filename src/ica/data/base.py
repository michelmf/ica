"""Base class for loading source data as torch Datasets."""
from abc import ABC, abstractmethod
from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset


class BaseDataset(Dataset[Tensor], ABC):
    """Base dataset class for multiple sources."""

    def __init__(self, source: Path) -> None:
        self.load(source)

    @abstractmethod
    def load(self, source: Path) -> None:
        """Load data from `source`, lazily or eagerly depending on the subclass."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> Tensor: ...
