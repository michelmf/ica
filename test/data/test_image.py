"""Tests for ica.data.image.ImageDataset.

Loading itself (file/column validation, __len__, __getitem__, to_tensor) is
inherited unchanged from TabularDataset and already covered by
test/data/test_tabular.py; only `to_image` is specific to this class.
"""

from pathlib import Path

import pandas as pd
import pytest
import torch

from ica.data.image import ImageDataset


def _write_csv(path: Path, columns: dict[str, list[float]]) -> None:
    pd.DataFrame(columns).to_csv(path, index=False)


class TestToImage:
    """Behavior of to_image(index)."""

    def test_reshapes_square_column_into_a_2d_image(self, tmp_path: Path) -> None:
        """A column with a perfect-square pixel count reshapes to (side, side)."""
        path = tmp_path / "mix.csv"
        # 4 pixels = a 2x2 image.
        _write_csv(
            path,
            {
                "mistura1": [0.0, 1.0, 2.0, 3.0],
                "mistura2": [3.0, 2.0, 1.0, 0.0],
            },
        )

        dataset = ImageDataset(path)
        image = dataset.to_image(0)

        assert image.shape == (2, 2)
        assert torch.allclose(image, torch.tensor([[0.0, 1.0], [2.0, 3.0]]))

    def test_non_square_column_raises_value_error_without_explicit_height(
        self, tmp_path: Path
    ) -> None:
        """A pixel count with no integer square root can't be reshaped blindly."""
        path = tmp_path / "mix.csv"
        _write_csv(
            path,
            {
                "mistura1": [0.0, 1.0, 2.0, 3.0, 4.0],
                "mistura2": [4.0, 3.0, 2.0, 1.0, 0.0],
            },
        )

        dataset = ImageDataset(path)

        with pytest.raises(ValueError, match="not a perfect square"):
            dataset.to_image(0)

    def test_non_square_column_reshapes_with_explicit_height(
        self, tmp_path: Path
    ) -> None:
        """An explicit height reshapes a non-square pixel count (e.g. 2x3)."""
        path = tmp_path / "mix.csv"
        _write_csv(
            path,
            {
                "mistura1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "mistura2": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            },
        )

        dataset = ImageDataset(path)
        image = dataset.to_image(0, height=2)

        assert image.shape == (2, 3)
        assert torch.allclose(image, torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))
