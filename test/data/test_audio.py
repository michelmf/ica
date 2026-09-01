"""Tests for ica.data.audio.AudioDataset."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from ica.data.audio import AudioDataset


def _write_wav(
    path: Path,
    data: np.ndarray,
    sample_rate: int = 16_000,
) -> None:
    sf.write(path, data, sample_rate, subtype="FLOAT")


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return tmp_path / "run1"


class TestInit:
    """Validation performed when constructing an AudioDataset from a directory."""

    def test_missing_directory_raises_file_not_found(self, run_dir: Path) -> None:
        """A path that doesn't exist on disk should fail fast with FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            AudioDataset(run_dir)

    def test_path_not_a_directory_raises_not_a_directory(self, tmp_path: Path) -> None:
        """Passing a file instead of a directory should raise NotADirectoryError."""
        file_path = tmp_path / "not_a_dir.wav"
        _write_wav(file_path, np.zeros((10, 1), dtype=np.float32))

        with pytest.raises(NotADirectoryError):
            AudioDataset(file_path)

    def test_empty_directory_raises_value_error(self, run_dir: Path) -> None:
        """A directory with no files at all has nothing to load."""
        run_dir.mkdir()

        with pytest.raises(ValueError, match="No WAV files"):
            AudioDataset(run_dir)

    def test_ignores_non_wav_files(self, run_dir: Path) -> None:
        """Non-.wav files in the directory should not count as audio sources."""
        run_dir.mkdir()
        (run_dir / "notes.txt").write_text("not audio")

        with pytest.raises(ValueError, match="No WAV files"):
            AudioDataset(run_dir)

    def test_single_file_raises_value_error(self, run_dir: Path) -> None:
        """ICA needs at least two sources, so a lone WAV file is rejected."""
        run_dir.mkdir()
        _write_wav(run_dir / "a.wav", np.zeros((10, 1), dtype=np.float32))

        with pytest.raises(ValueError, match="at least 2"):
            AudioDataset(run_dir)

    def test_mismatched_sample_rates_raises_value_error(self, run_dir: Path) -> None:
        """All sources in a run must share a sample rate to be comparable."""
        run_dir.mkdir()
        _write_wav(
            run_dir / "a.wav", np.zeros((10, 1), dtype=np.float32), sample_rate=16_000
        )
        _write_wav(
            run_dir / "b.wav", np.zeros((10, 1), dtype=np.float32), sample_rate=44_100
        )

        with pytest.raises(ValueError, match="same sample rate"):
            AudioDataset(run_dir)

    def test_valid_directory_loads_sorted_files(self, run_dir: Path) -> None:
        """A valid run directory yields files in sorted order and the shared sample rate."""
        run_dir.mkdir()
        _write_wav(
            run_dir / "b.wav", np.zeros((10, 1), dtype=np.float32), sample_rate=16_000
        )
        _write_wav(
            run_dir / "a.wav", np.zeros((10, 1), dtype=np.float32), sample_rate=16_000
        )

        dataset = AudioDataset(run_dir)

        assert [f.name for f in dataset.files] == ["a.wav", "b.wav"]
        assert dataset.sample_rate == 16_000


class TestLen:
    """Behavior of len(dataset)."""

    def test_returns_number_of_files(self, run_dir: Path) -> None:
        """len() should reflect how many WAV sources were found."""
        run_dir.mkdir()
        _write_wav(run_dir / "a.wav", np.zeros((10, 1), dtype=np.float32))
        _write_wav(run_dir / "b.wav", np.zeros((10, 1), dtype=np.float32))
        _write_wav(run_dir / "c.wav", np.zeros((10, 1), dtype=np.float32))

        dataset = AudioDataset(run_dir)

        assert len(dataset) == 3


class TestGetItem:
    """Behavior of dataset[index], including the mono conversion."""

    def test_averages_multichannel_audio_into_mono(self, run_dir: Path) -> None:
        """
        Multichannel audio should be collapsed to mono by averaging
        channels per frame.
        """
        run_dir.mkdir()
        stereo = np.array(
            [[0.0, 1.0], [0.5, -0.5], [0.25, 0.75], [-1.0, 1.0]],
            dtype=np.float32,
        )
        _write_wav(run_dir / "a.wav", stereo)
        _write_wav(run_dir / "b.wav", np.zeros((4, 2), dtype=np.float32))

        dataset = AudioDataset(run_dir)
        waveform = dataset[0]

        expected = torch.tensor([0.5, 0.0, 0.5, 0.0])
        assert waveform.shape == (4,)
        assert torch.allclose(waveform, expected, atol=1e-4)

    def test_preserves_mono_audio(self, run_dir: Path) -> None:
        """A single-channel source should pass through unchanged."""
        run_dir.mkdir()
        mono = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        _write_wav(run_dir / "a.wav", mono)
        _write_wav(run_dir / "b.wav", np.zeros(4, dtype=np.float32))

        dataset = AudioDataset(run_dir)
        waveform = dataset[0]

        assert waveform.shape == (4,)
        assert torch.allclose(waveform, torch.from_numpy(mono), atol=1e-4)

    def test_sample_rate_mismatch_at_read_time_raises_runtime_error(
        self, run_dir: Path
    ) -> None:
        """If a source's sample rate no longer matches the dataset's, reading it should fail loudly."""
        run_dir.mkdir()
        _write_wav(run_dir / "a.wav", np.zeros((10, 1), dtype=np.float32))
        _write_wav(run_dir / "b.wav", np.zeros((10, 1), dtype=np.float32))

        dataset = AudioDataset(run_dir)
        dataset.sample_rate = 8_000

        with pytest.raises(RuntimeError, match="Unexpected sample rate"):
            dataset[0]
