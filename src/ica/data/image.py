"""Dataset for loading image data used in ICA computation."""
from math import isqrt

from torch import Tensor

from ica.data.tabular import TabularDataset


class ImageDataset(TabularDataset):
    """
    Dataset containing mixture columns from a single CSV file, one column
    per flattened source/mixture image (Example 12.3, p.258). Loading is
    identical to `TabularDataset`; the only addition is reshaping a column
    back into a 2D image for display.
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
