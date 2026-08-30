"""Dataset for loading audio data used in ICA computation."""
from pathlib import Path

import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset


class AudioDataset(Dataset[Tensor]):
    """Dataset containing WAV audio sources from a single run."""
    def __init__(self, path: Path) -> None:
        self.path = path

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

    def _validate_sample_rates(self) -> int:
        """Ensure that every audio source uses the same sample rate."""

        sample_rates = {sf.info(file).samplerate for file in self.files}

        if len(sample_rates) != 1:
            raise ValueError(
                "All WAV files in a run must have the same sample rate. "
                f"Found: {sorted(sample_rates)}"
            )
        return sample_rates.pop()
