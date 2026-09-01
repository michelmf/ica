# TODO

## Reporting results — needs a decision first

`ica.report` is still just a docstring — nothing is implemented, and before
writing code it's worth deciding what "reporting a run" actually means here.
Candidate pieces, from the book's own vocabulary:

- **Convergence** — per-unit iteration count vs. `max_iter`, and the value of
  the contrast (`|kurt(wᵀz)|` or the negentropy approximation `J`) at each
  iteration, so a convergence curve like Figs. 8.13/8.17 can be plotted.
  Currently `FixedPointICA` doesn't record this at all (`fit` only keeps the
  final `components`); it would need to optionally log the trajectory.
- **Mixing matrix quality** — `local/separate_audio.py`, `local/separate_tabular.py`,
  and `local/separate_images.py` already recover
  `A = V⁻¹Wᵀ` and cross-checks Kurtosis vs. Negentropy by aligning columns
  (permutation + sign) and taking a max abs difference. For synthetic data
  with a known `A`, the same alignment could score against ground truth
  (e.g. an Amari-index-style metric) instead of cross-technique agreement.
- **Source quality** — with ground-truth sources (synthetic tests): best-match
  correlation, already used in `test/model/_ica.py`
  (`best_match_correlations`); could add SIR/SDR for a stricter number. For
  real audio (no ground truth, e.g. the `dataset/audio/run*` runs): no
  reference exists, so quality has to fall back to something reference-free —
  final contrast value, or just listening to the separated WAVs.
- **Format** — is this a notebook, a script that dumps plots/tables, or
  actual `ica.report` code with a defined API (e.g. `Report.from_model(model,
  X)`)? Also whether it needs to run per-dataset-run (8 audio runs exist) or
  just on demand.

None of this is implemented yet — this section exists to pin down scope
before writing `ica.report`.

## Data (`ica.data`)

- [x] `ImageDataset` — subclasses `TabularDataset` (same CSV-columns-as-sources
      loading), adds `to_image(index, height=None)` to reshape a flattened
      column back to 2D (auto-detects a square side via `isqrt`, or takes an
      explicit `height`). Tested in `test/data/test_image.py`.
  - [ ] **RGB channel layout is unknown and unimplemented.** `run3/mix_imagens_rgb.csv`
        has 9 columns; each is currently treated as an independent
        single-channel image (same as grayscale). Nobody has confirmed
        whether that's "3 images x 3 channels (R,G,B)" or something else, so
        there's no code to regroup 3 columns into one color image yet — do
        that once the convention is confirmed.
  - [ ] **Run layout doesn't match the assignment.** The brief (Example 12.3,
        p.258) calls for run1=2 sources, run2=3, run3=4 (9 images total across
        3 runs). What's on disk is `run1` (3 columns, grayscale) and `run3`
        (9 columns, RGB) — no `run2`, and neither column count matches the
        brief. Unclear whether the dataset folder is unfinished or the
        run-number mapping is just different from the brief's; needs
        clarifying before treating any run's source count as meaningful.
- [x] `TabularDataset` — one CSV file per run (`dataset/tabular/run*/mix_*_stats.csv`),
      one column per source, tested in `test/data/test_tabular.py`.
  - [ ] **`run8` produces NaN components.** `mistura2` ranges from ~1 to
        ~1.17e12 (0.01% of rows exceed 1e9), so its variance dominates the
        covariance matrix by ~19 orders of magnitude over the other columns.
        The resulting condition number (~1e19) exceeds even float64
        precision, so `Whitening`'s `eigh` returns slightly-negative "noise"
        eigenvalues and `rsqrt()` turns them into NaN — confirmed by
        reproducing it manually (`torch.linalg.eigh` on `run8`'s centered
        covariance). Not a code bug: no numeric precision fixes this once the
        condition number is this extreme. Possibly a deliberate edge case for
        Section 8.3.1's "kurtosis is sensitive to outliers" critique, but
        that critique is about the contrast function, not about whitening
        itself blowing up — worth asking whether `run8` is meant to be
        solvable as-is, or needs a robustness step before whitening (e.g.
        per-column log/robust scaling) that isn't part of the book's method.
        `local/separate_tabular.py` detects and warns about this rather than
        silently writing garbage output.

## Preprocessing (`ica.preprocessing`)

- [ ] Per-modality preprocessing for audio (`preprocessing/audio.py` is an
      empty stub) — anything that needs to happen before centering/whitening
      (e.g. ensuring equal length between sources).
- [ ] Same for tabular/image (`preprocessing/tabular.py`, `preprocessing/image.py`,
      both empty stubs) — nothing modality-specific needed yet for either
      (a CSV's columns already share one row count).
- [ ] No direct unit tests for `Centering`/`Whitening` (`preprocessing/common.py`)
      — currently only exercised indirectly through the model tests.

## Model (`ica.model`)

- [x] `Kurtosis` (Section 8.2.3, fixed-point).
- [x] `Negentropy` (Section 8.3.5, fixed-point; `logcosh`/`gauss`/`cube`
      nonlinearities, 8.31-8.33).
- [x] Deflationary orthogonalization (8.4.2) and symmetric orthogonalization
      (8.4.3), both behind `FixedPointICA(orthogonalization=...)`.
- [ ] Gradient algorithms (8.2.2, 8.3.4) — skipped so far in favor of the
      fixed-point versions; not planned unless there's a reason to want the
      online/adaptive variant.
- [ ] Section 8.5 (ICA and projection pursuit) and anything past Chapter 8 —
      not started.
- [ ] No direct unit test for `ica.utils.Pipeline` itself (only exercised
      indirectly through the model tests).

## Tooling

- [ ] `pyproject.toml` declares `[project.scripts] ica = "ica:main"`, but
      `ica/__init__.py` has no `main` — `uv run ica` would fail right now.
- [ ] `local/pipeline_example.py` still defines its own mock
      `CenterPreprocessor`/`NormalizePreprocessor` classes, predating
      `preprocessing.common.Centering`/`Whitening` — worth swapping to the
      real ones, or deleting the file now that `local/separate_audio.py`
      supersedes it.
