"""ICA by maximization of negentropy (Section 8.3)."""
from typing import Literal

import torch
from torch import Tensor

from ica.model.fixed_point import FixedPointICA, Orthogonalization

Nonlinearity = Literal["logcosh", "gauss", "cube"]


class Negentropy(FixedPointICA):
    """
    Estimates independent components from whitened data by maximizing an
    approximation of negentropy, one unit at a time (Section 8.3.5).
    """

    def __init__(
        self,
        nonlinearity: Nonlinearity = "logcosh",
        a1: float = 1.0,
        n_components: int | None = None,
        max_iter: int = 200,
        tol: float = 1e-6,
        orthogonalization: Orthogonalization = "deflation",
    ) -> None:
        super().__init__(
            n_components=n_components,
            max_iter=max_iter,
            tol=tol,
            orthogonalization=orthogonalization,
        )
        self.nonlinearity = nonlinearity
        self.a1 = a1

    def _update(self, w: Tensor, X: Tensor) -> Tensor:
        y = w @ X
        g, g_prime = self._g(y)
        # w <- E{z g(w^T z)} - E{g'(w^T z)} w, equation 8.43
        return (X * g).mean(dim=1) - g_prime.mean() * w

    def _g(self, y: Tensor) -> tuple[Tensor, Tensor]:
        if self.nonlinearity == "logcosh":
            # g_1(y) = tanh(a1 y), equation 8.31
            g = torch.tanh(self.a1 * y)
            return g, self.a1 * (1 - g**2)
        if self.nonlinearity == "gauss":
            # g_2(y) = y exp(-y^2/2), equation 8.32
            exp = torch.exp(-(y**2) / 2)
            return y * exp, (1 - y**2) * exp
        if self.nonlinearity == "cube":
            # g_3(y) = y^3, equation 8.33
            return y**3, 3 * y**2
        raise ValueError(f"Unknown nonlinearity: {self.nonlinearity!r}")
