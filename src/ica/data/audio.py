"""Dataset for loading audio data used in ICA computation."""
from pathlib import Path

import soundfile as sf
import torch
from torch import Tensor

from ica.data.base import BaseDataset


class AudioDataset(BaseDataset):
    """Dataset containing WAV audio sources from a single run."""

    def load(self, source: Path) -> None:
        """Index the WAV files in `source` (lazily, no audio is read yet)."""
        self.path = source

        if not self.path.exists():
            raise FileNotFoundError(
                f"Audio directory does not exist: {self.path}"
            )
        if not self.path.is_dir():
            raise NotADirectoryError(
                f"Expected a directory, received: {self.path}"
            )
        self.files = sorted(self.path.glob("*.wav"))

        if not self.files:
            raise ValueError(
                f"No WAV files were found in: {self.path}"
            )
        if len(self.files) < 2:
            raise ValueError(
                f"Expected at least 2 WAV files in the directory, "
                f"found {len(self.files)}."
            )
        self.sample_rate = self._validate_sample_rates()

    def __repr__(self) -> str:
        n_samples = sf.info(self.files[0]).frames
        alias = f"alias={self.alias!r}, " if self.alias is not None else ""
        return (
            f"{type(self).__name__}({alias}path={self.path!r}, "
            f"X_shape=({len(self.files)}, {n_samples}), "
            f"sample_rate={self.sample_rate})"
        )

    def __len__(self) -> int:
        """Return the number of audio sources."""
        return len(self.files)

    def __getitem__(self, index: int) -> Tensor:
        """Load one audio source as a tensor."""
        audio, sample_rate = sf.read(
            self.files[index],
            dtype="float32",
            always_2d=True,
        )

        if sample_rate != self.sample_rate:
            raise RuntimeError(
                f"Unexpected sample rate in {self.files[index]}"
            )
        # Convert stereo multichannel to mono.
        waveform = torch.from_numpy(audio).transpose(0, 1)
        waveform = waveform.mean(dim=0)

        return waveform

    def to_tensor(self) -> Tensor:
        """Stack every audio source into the mixture matrix X (n_sources, n_samples)."""
        return torch.stack([self[index] for index in range(len(self))], dim=0)

    def _validate_sample_rates(self) -> int:
        """Ensure that every audio source uses the same sample rate."""

        sample_rates = {sf.info(file).samplerate for file in self.files}

        if len(sample_rates) != 1:
            raise ValueError(
                "All WAV files in a run must have the same sample rate. "
                f"Found: {sorted(sample_rates)}"
            )
        return sample_rates.pop()
