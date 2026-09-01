"""Dataset for loading tabular data used in ICA computation."""
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from ica.data.base import BaseDataset


class TabularDataset(BaseDataset):
    """Dataset containing mixture columns from a single CSV file."""

    def load(self, source: Path) -> None:
        """Load the CSV at `source` eagerly; each column is one mixture source."""
        self.path = source

        if not self.path.exists():
            raise FileNotFoundError(
                f"Tabular file does not exist: {self.path}"
            )
        if not self.path.is_file():
            raise IsADirectoryError(
                f"Expected a file, received: {self.path}"
            )

        self.data = pd.read_csv(self.path)

        if self.data.shape[1] < 2:
            raise ValueError(
                f"Expected at least 2 mixture columns in {self.path}, "
                f"found {self.data.shape[1]}."
            )

    def __repr__(self) -> str:
        alias = f"alias={self.alias!r}, " if self.alias is not None else ""
        return (
            f"{type(self).__name__}({alias}path={self.path!r}, "
            f"X_shape=({len(self)}, {self.data.shape[0]}))"
        )

    def __len__(self) -> int:
        """Return the number of mixture columns (sources)."""
        return self.data.shape[1]

    def __getitem__(self, index: int) -> Tensor:
        """Load one mixture column as a tensor."""
        column = self.data.iloc[:, index].to_numpy(dtype="float32")
        return torch.from_numpy(column)

    def to_tensor(self) -> Tensor:
        """Stack every mixture column into the mixture matrix X (n_sources, n_samples)."""
        return torch.stack([self[index] for index in range(len(self))], dim=0)
