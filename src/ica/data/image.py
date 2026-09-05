"""Dataset for loading image data used in ICA computation."""
from math import isqrt

from torch import Tensor

from ica.data.tabular import TabularDataset


class ImageDataset(TabularDataset):
    """
    Dataset for image mixtures, where each column in the CSV represents a 
    single flattened image. Inherits data loading from `TabularDataset`, 
    with the added capability to reshape columns back into 2D images for 
    visualization or further processing.
    """

    def to_image(self, index: int, height: int | None = None) -> Tensor:
        """
        Reshape column `index` back into a 2D image. The CSV stores no
        dimensions, so a square image is assumed unless `height` is given
        explicitly.
        """
        column = self[index]
        n_pixels = len(column)

        if height is None:
            height = isqrt(n_pixels)
            if height * height != n_pixels:
                raise ValueError(
                    f"Column {index} has {n_pixels} pixels, which is not a "
                    f"perfect square; pass `height` explicitly."
                )
            return column.reshape(height, height)

        if n_pixels % height != 0:
            raise ValueError(
                f"Column {index} has {n_pixels} pixels, not divisible by "
                f"height={height}."
            )
        return column.reshape(height, n_pixels // height)
