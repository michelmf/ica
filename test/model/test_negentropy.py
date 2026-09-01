"""Tests for ica.model.negentropy.Negentropy."""

import pytest
import torch
from torch.distributions import Laplace, Uniform

from ica.model.negentropy import Negentropy

from _ica import best_match_correlations, mix, separate, N_SAMPLES


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.mark.parametrize("orthogonalization", ["deflation", "symmetric"])
@pytest.mark.parametrize("nonlinearity", ["logcosh", "gauss", "cube"])
class TestSeparation:
    """Negentropy should recover each source, for every nonlinearity (8.31-8.33)
    and every orthogonalization strategy (8.4.2, 8.4.3), regardless of the
    distribution family of the sources.
    """

    def test_two_distinct_uniform_sources(
        self, nonlinearity: str, orthogonalization: str
    ) -> None:
        """Two subgaussian sources with different ranges should still separate."""
        s1 = Uniform(-1.0, 1.0).sample((N_SAMPLES,))
        s2 = Uniform(-3.0, 3.0).sample((N_SAMPLES,))
        X, S = mix(s1, s2)

        model = Negentropy(nonlinearity=nonlinearity, orthogonalization=orthogonalization)
        Y = separate(model, X)

        assert (best_match_correlations(Y, S) > 0.95).all()

    def test_two_laplacian_sources(
        self, nonlinearity: str, orthogonalization: str
    ) -> None:
        """Two supergaussian sources with different scales should still separate."""
        s1 = Laplace(0.0, 1.0).sample((N_SAMPLES,))
        s2 = Laplace(0.0, 2.0).sample((N_SAMPLES,))
        X, S = mix(s1, s2)

        model = Negentropy(nonlinearity=nonlinearity, orthogonalization=orthogonalization)
        Y = separate(model, X)

        assert (best_match_correlations(Y, S) > 0.95).all()

    def test_uniform_and_laplacian_sources(
        self, nonlinearity: str, orthogonalization: str
    ) -> None:
        """A subgaussian and a supergaussian source mixed together should separate."""
        s1 = Uniform(-1.0, 1.0).sample((N_SAMPLES,))
        s2 = Laplace(0.0, 1.0).sample((N_SAMPLES,))
        X, S = mix(s1, s2)

        model = Negentropy(nonlinearity=nonlinearity, orthogonalization=orthogonalization)
        Y = separate(model, X)

        assert (best_match_correlations(Y, S) > 0.95).all()


class TestUnknownNonlinearity:
    """An invalid nonlinearity name should fail loudly, not silently no-op."""

    def test_raises_value_error(self) -> None:
        s1 = Uniform(-1.0, 1.0).sample((N_SAMPLES,))
        s2 = Uniform(-3.0, 3.0).sample((N_SAMPLES,))
        X, _ = mix(s1, s2)

        with pytest.raises(ValueError, match="Unknown nonlinearity"):
            separate(Negentropy(nonlinearity="not-a-nonlinearity"), X)
