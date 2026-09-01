"""Tests for ica.data.tabular.TabularDataset."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from ica.data.tabular import TabularDataset


def _write_csv(path: Path, columns: dict[str, list[float]]) -> None:
    pd.DataFrame(columns).to_csv(path, index=False)


class TestInit:
    """Validation performed when constructing a TabularDataset from a CSV file."""

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """A path that doesn't exist on disk should fail fast with FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TabularDataset(tmp_path / "missing.csv")

    def test_path_is_a_directory_raises_is_a_directory_error(
        self, tmp_path: Path
    ) -> None:
        """Passing a directory instead of a file should raise IsADirectoryError."""
        with pytest.raises(IsADirectoryError):
            TabularDataset(tmp_path)

    def test_single_column_raises_value_error(self, tmp_path: Path) -> None:
        """ICA needs at least two sources, so a single-column file is rejected."""
        path = tmp_path / "mix.csv"
        _write_csv(path, {"mistura1": [0.0, 1.0, 2.0]})

        with pytest.raises(ValueError, match="at least 2"):
            TabularDataset(path)

    def test_valid_file_loads(self, tmp_path: Path) -> None:
        """A valid CSV file loads without error."""
        path = tmp_path / "mix.csv"
        _write_csv(path, {"mistura1": [0.0, 1.0], "mistura2": [2.0, 3.0]})

        dataset = TabularDataset(path, alias="run1")

        assert dataset.path == path
        assert dataset.alias == "run1"


class TestLen:
    """Behavior of len(dataset)."""

    def test_returns_number_of_columns(self, tmp_path: Path) -> None:
        """len() should reflect how many mixture columns were found."""
        path = tmp_path / "mix.csv"
        _write_csv(
            path,
            {"mistura1": [0.0, 1.0], "mistura2": [2.0, 3.0], "mistura3": [4.0, 5.0]},
        )

        dataset = TabularDataset(path)

        assert len(dataset) == 3


class TestGetItem:
    """Behavior of dataset[index]."""

    def test_returns_column_as_tensor(self, tmp_path: Path) -> None:
        """Each index should return the corresponding CSV column as a float tensor."""
        path = tmp_path / "mix.csv"
        _write_csv(
            path, {"mistura1": [0.1, -0.2, 0.3], "mistura2": [1.0, 2.0, 3.0]}
        )

        dataset = TabularDataset(path)

        assert torch.allclose(
            dataset[0], torch.tensor([0.1, -0.2, 0.3]), atol=1e-6
        )
        assert torch.allclose(
            dataset[1], torch.tensor([1.0, 2.0, 3.0]), atol=1e-6
        )


class TestToTensor:
    """Behavior of to_tensor()."""

    def test_stacks_columns_into_mixture_matrix(self, tmp_path: Path) -> None:
        """to_tensor() should stack columns into an (n_sources, n_samples) matrix."""
        path = tmp_path / "mix.csv"
        _write_csv(
            path, {"mistura1": [0.1, -0.2, 0.3], "mistura2": [1.0, 2.0, 3.0]}
        )

        dataset = TabularDataset(path)
        X = dataset.to_tensor()

        assert X.shape == (2, 3)
        assert torch.allclose(X[0], torch.tensor([0.1, -0.2, 0.3]), atol=1e-6)
        assert torch.allclose(X[1], torch.tensor([1.0, 2.0, 3.0]), atol=1e-6)
