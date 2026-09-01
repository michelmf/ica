"""Tests for ica.model.kurtosis.Kurtosis."""

import pytest
import torch
from torch.distributions import Laplace, Uniform

from ica.model.kurtosis import Kurtosis

from _ica import best_match_correlations, mix, separate, N_SAMPLES


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.mark.parametrize("orthogonalization", ["deflation", "symmetric"])
class TestSeparation:
    """Kurtosis should recover each source regardless of its distribution family."""

    def test_two_distinct_uniform_sources(self, orthogonalization: str) -> None:
        """Two subgaussian sources with different ranges should still separate."""
        s1 = Uniform(-1.0, 1.0).sample((N_SAMPLES,))
        s2 = Uniform(-3.0, 3.0).sample((N_SAMPLES,))
        X, S = mix(s1, s2)

        Y = separate(Kurtosis(orthogonalization=orthogonalization), X)

        assert (best_match_correlations(Y, S) > 0.95).all()

    def test_two_laplacian_sources(self, orthogonalization: str) -> None:
        """Two supergaussian sources with different scales should still separate."""
        s1 = Laplace(0.0, 1.0).sample((N_SAMPLES,))
        s2 = Laplace(0.0, 2.0).sample((N_SAMPLES,))
        X, S = mix(s1, s2)

        Y = separate(Kurtosis(orthogonalization=orthogonalization), X)

        assert (best_match_correlations(Y, S) > 0.95).all()

    def test_uniform_and_laplacian_sources(self, orthogonalization: str) -> None:
        """A subgaussian and a supergaussian source mixed together should separate."""
        s1 = Uniform(-1.0, 1.0).sample((N_SAMPLES,))
        s2 = Laplace(0.0, 1.0).sample((N_SAMPLES,))
        X, S = mix(s1, s2)

        Y = separate(Kurtosis(orthogonalization=orthogonalization), X)

        assert (best_match_correlations(Y, S) > 0.95).all()
