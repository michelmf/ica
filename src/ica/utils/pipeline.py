"""Generic pipeline chaining preprocessing and ICA operations."""
from typing import Any, Self

from torch import Tensor

from ica.data.base import BaseDataset


class Pipeline:
    """
    Chains an arbitrary sequence of operations applied to a tensor.

    Every step exposes `fit(X) -> Self` and `transform(X) -> Tensor`
    (preprocessors default `fit` to a no-op, ICA models implement it for
    real), so the pipeline treats all steps the same way. Steps are
    optional: an empty pipeline just returns `dataset.to_tensor()`.
    """

    def __init__(self, steps: list[Any] | None = None) -> None:
        self.steps = steps or []

    def fit(self, dataset: BaseDataset) -> Self:
        """Fit every step in sequence, forwarding each one's output to the next."""
        self.fit_transform(dataset)
        return self

    def transform(self, dataset: BaseDataset) -> Tensor:
        """Apply every already-fitted step in sequence."""
        X = dataset.to_tensor()
        for step in self.steps:
            X = step.transform(X)
        return X

    def fit_transform(self, dataset: BaseDataset) -> Tensor:
        """Fit and apply every step in sequence."""
        X = dataset.to_tensor()
        for step in self.steps:
            step.fit(X)
            X = step.transform(X)
        return X

    # TODO: add __add__ behavior