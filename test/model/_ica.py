"""Shared helpers for testing ICA models (not a test module itself)."""

import torch
from torch import Tensor

from ica.model.base import BaseICA
from ica.preprocessing.common import Centering, Whitening

N_SAMPLES = 20_000
MIXING_MATRIX = torch.tensor([[1.0, 0.8], [0.5, 1.2]])


def mix(s1: Tensor, s2: Tensor) -> tuple[Tensor, Tensor]:
    """Stack two independent sources and mix them with a fixed, non-orthogonal matrix."""
    S = torch.stack([s1, s2], dim=0)
    return MIXING_MATRIX @ S, S


def separate(model: BaseICA, X: Tensor) -> Tensor:
    """Center, whiten, and recover independent components from a mixture."""
    Xc = Centering().fit(X).transform(X)
    Xw = Whitening().fit(Xc).transform(Xc)
    return model.fit_transform(Xw)


def best_match_correlations(Y: Tensor, S: Tensor) -> Tensor:
    """
    For each recovered component, the absolute correlation with the source it
    matches best. Recovered components can come back in any order and with a
    flipped sign, so it's this pairing, not the raw correlation matrix, that
    should look like a permutation.
    """
    Yn = (Y - Y.mean(dim=1, keepdim=True)) / Y.std(dim=1, keepdim=True)
    Sn = (S - S.mean(dim=1, keepdim=True)) / S.std(dim=1, keepdim=True)
    corr = (Yn @ Sn.T / Y.shape[1]).abs()
    return corr.amax(dim=1)
