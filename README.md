# ica

Independent Component Analysis toolkit, implemented from the theory in Hyvärinen,
Karhunen & Oja (2001). Built as coursework for the ICA class. Not intended as a
general-purpose package.

## Overview

ICA recovers a set of statistically independent source signals from observed
mixtures of them (e.g. separating overlapping voices recorded by several
microphones). This project provides the building blocks for that pipeline:

- **`ica.data`** — dataset loaders that expose a mixture matrix `X` of shape
  `(n_sources, n_samples)` via `to_tensor()`. `AudioDataset` (WAV sources, one
  file per source), `TabularDataset` (CSV file, one column per source), and
  `ImageDataset` (same as tabular, plus `to_image()` to reshape a flattened
  column back to 2D) are all implemented.
- **`ica.preprocessing`** — steps shared across modalities, `Centering` and
  `Whitening`, plus per-modality preprocessing for anything that needs to
  happen before them (framing, resampling, etc.).
- **`ica.model`** — `BaseICA`, the abstract class concrete ICA algorithms
  implement (`fit`/`transform`). `Kurtosis` (Section 8.2) and `Negentropy`
  (Section 8.3) estimate independent components via a one-unit fixed-point
  algorithm, extracting several components either by deflationary (8.4.2,
  default) or symmetric (8.4.3) orthogonalization.
- **`ica.utils.Pipeline`** — chains preprocessing and model steps, each
  exposing `fit`/`transform`, into a single `fit_transform` call.
- **`ica.report`** — result reporting/visualization (not yet implemented).

## Project status

Work in progress. Audio, tabular, and image datasets all load and run through
the full pipeline (dataset + preprocessing + kurtosis/negentropy models), with
test coverage; see `todo.md` for open questions (notably reporting, and the
image dataset's RGB channel layout and run-count mismatch with the brief).

## Installation

Requires Python >=3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

```python
from pathlib import Path

from ica.data.audio import AudioDataset
from ica.model import Kurtosis
from ica.preprocessing import Centering, Whitening
from ica.utils import Pipeline

dataset = AudioDataset(source=Path("dataset/audio/run1"), alias="run1")

pipeline = Pipeline(steps=[Centering(), Whitening(), Kurtosis()])
Y = pipeline.fit_transform(dataset)  # separated independent components
```

## Testing

```bash
uv run pytest
```

## Reference

Hyvärinen, A., Karhunen, J., & Oja, E. (2001). *Independent Component
Analysis*. John Wiley & Sons.
