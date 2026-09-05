"""
Shared one-unit fixed-point scaffold for Chapter 8 methods (kurtosis,
negentropy): estimate a component at a time via a fixed-point contrast
update, decorrelating units by deflationary or symmetric orthogonalization.
"""
from abc import ABC, abstractmethod
from typing import Literal, Self

import torch
from torch import Tensor

from ica.model.base import BaseICA

Orthogonalization = Literal["deflation", "symmetric"]


def _symmetric_orthogonalize(W: Tensor) -> Tensor:
    """(WW^T)^(-1/2) W, equations 8.48-8.49."""
    eigvals, eigvecs = torch.linalg.eigh(W @ W.T)
    return eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T @ W


class FixedPointICA(BaseICA, ABC):
    """
    Extracts several independent components from whitened data by running a
    one-unit fixed-point algorithm, then decorrelating the estimated units
    either one at a time by deflationary orthogonalization (Section 8.4.2,
    the default), or all at once by symmetric orthogonalization
    (Section 8.4.3). Subclasses only need to provide the one-unit update
    rule for a single contrast function (e.g. kurtosis, negentropy).
    """

    def __init__(
        self,
        n_components: int | None = None,
        max_iter: int = 200,
        tol: float = 1e-6,
        orthogonalization: Orthogonalization = "deflation",
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.orthogonalization = orthogonalization

    @abstractmethod
    def _update(self, w: Tensor, X: Tensor) -> Tensor:
        """One fixed-point update step for a single unit w."""

    def fit(self, X: Tensor) -> Self:
        if self.orthogonalization == "deflation":
            self.components = self._fit_deflation(X)
        elif self.orthogonalization == "symmetric":
            self.components = self._fit_symmetric(X)
        else:
            raise ValueError(
                f"Unknown orthogonalization: {self.orthogonalization!r}"
            )
        return self

    def _fit_deflation(self, X: Tensor) -> Tensor:
        n_features, _ = X.shape
        n_components = self.n_components or n_features
        W = torch.zeros(n_components, n_features)

        for p in range(n_components):
            w = torch.randn(n_features)
            w = w / w.norm()

            for _ in range(self.max_iter):
                w_new = self._update(w, X)
                # w_p <- w_p - sum_j (w_p^T w_j) w_j, equation 8.47
                w_new = w_new - W[:p].T @ (W[:p] @ w_new)
                w_new = w_new / w_new.norm()

                converged = (w_new @ w).abs() > 1 - self.tol
                w = w_new
                if converged:
                    break

            W[p] = w

        return W

    def _fit_symmetric(self, X: Tensor) -> Tensor:
        n_features, _ = X.shape
        n_components = self.n_components or n_features

        W = _symmetric_orthogonalize(torch.randn(n_components, n_features))

        for _ in range(self.max_iter):
            W_new = torch.stack([self._update(w, X) for w in W])
            W_new = _symmetric_orthogonalize(W_new)

            converged = bool(((W_new * W).sum(dim=1).abs() > 1 - self.tol).all())
            W = W_new
            if converged:
                break

        return W

    def transform(self, X: Tensor) -> Tensor:
        return self.components @ X
