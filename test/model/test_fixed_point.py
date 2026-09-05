"""Tests for ica.model.fixed_point.FixedPointICA."""

import pytest
import torch

from ica.model.kurtosis import Kurtosis


class TestUnknownOrthogonalization:
    """An invalid orthogonalization name should fail loudly, not silently no-op."""

    def test_raises_value_error(self) -> None:
        X = torch.randn(2, 100)

        with pytest.raises(ValueError, match="Unknown orthogonalization"):
            Kurtosis(orthogonalization="not-a-strategy").fit(X)
