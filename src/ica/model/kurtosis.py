"""ICA by maximization of kurtosis (Section 8.2)."""
from torch import Tensor

from ica.model.base import FixedPointICA


class Kurtosis(FixedPointICA):
    """
    Estimates independent components from whitened data by maximizing the
    absolute value of kurtosis, one unit at a time (Section 8.2.3).
    """

    def _update(self, w: Tensor, X: Tensor) -> Tensor:
        # w <- E{z(w^T z)^3} - 3w, equation 8.20
        return (X * (w @ X) ** 3).mean(dim=1) - 3 * w
