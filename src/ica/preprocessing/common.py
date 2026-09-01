"""Preprocessing operations shared across every data modality."""
from typing import Self

import torch
from torch import Tensor

from ica.preprocessing.base import BasePreprocessor


class Centering(BasePreprocessor):
    """Subtracts the mean of each source (row) from X."""

    def fit(self, X: Tensor) -> Self:
        # x = x' - E{x'}, equation 7.10
        self.mean = X.mean(dim=1, keepdim=True)
        return self

    def transform(self, X: Tensor) -> Tensor:
        return X - self.mean


class Whitening(BasePreprocessor):
    """Decorrelates sources (rows) via eigendecomposition of the covariance matrix."""

    def fit(self, X: Tensor) -> Self:
        cov = torch.cov(X)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        # ED^(-1/2)E^T, equation 7.20
        self.whitening_matrix = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T
        return self

    def transform(self, X: Tensor) -> Tensor:
        return self.whitening_matrix @ X
