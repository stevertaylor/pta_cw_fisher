# PTA Continuous-Wave Fisher Analysis

This repository explores sky-localization precision for continuous gravitational-wave (CW) sources in pulsar timing arrays (PTAs).
The core approach evaluates **Fisher-matrix approximations** of the CW likelihood, implemented in JAX for automatic differentiation and GPU-friendly vectorization.
Code heritage traces to `gabefreedman/etudes`.

## Quick Start

The **recommended entry point** is:

```
ring_pterm_vectorized_fixed.ipynb   ← production notebook
```

It relies on:

| File | Role |
|------|------|
| `utils_vectorized_fixed.py` | JAX-vectorized signal model & Fisher routines (`PulsarBatch`, `get_delay_batch`, `compute_total_fisher`, etc.) |
| `constants.py` | Physical constants (c, pc, GMsun, …) shared by all modules |
| `omega_interpolation_data.npz` | Cached interpolation tables used by several utilities |

### Fixes relative to `ring_pterm_vectorized.ipynb`

`utils_vectorized_fixed.py` corrects two physics issues found in `utils_vectorized.py`:

1. **`omega_p0` consistency** — the pulsar-term reference frequency is now computed analytically in *both* `get_delay_batch` (phase-free branch) and `compute_pulsar_phases_batch`, removing a subtle mismatch.
2. **`tref` parameter** — a reference-time argument has been threaded through `get_delay_batch`, `computer_snr2_batch`, and `compute_total_fisher`, restoring full parity with the `CW_Signal` class in `utils.py`.

Figures produced by the fixed notebook are saved to `outputs_fixed/`.

---

## Studies in the Notebook

The notebook is organized as a series of self-contained studies.

| Study | Title | What it explores |
|-------|-------|------------------|
| 0 | Single Ring Configuration | Places a ring of pulsars at a fixed angular radius around the GW source; plots antennapattern geometry, timing residuals, and the Fisher-matrix localization ellipse. |
| 1 | Sky Localization vs Pulsar–Source Separation | Sweeps angular radius of the pulsar ring; tracks how ΔΩ varies with separation. |
| 1a | Distance Uncertainty Dependence | Same sweep for multiple distance-precision levels (fractions of the GW wavelength). |
| 2 | All-Sky Localization Map | Fixes the pulsar ring around the North Pole, sweeps GW source across the sky (HEALPix), producing all-sky ΔΩ maps. |
| 3 | Linked vs Decoupled Phase | Compares localization when the pulsar-term phase is physically determined ("linked / interferometric") vs treated as a free parameter ("decoupled"). |
| 4 | Distance Uncertainty — Linked Case | Sweeps distance uncertainty in the linked regime to map the transition from precise to uninformative distance knowledge. |
| 5 | Distance Uncertainty — Decoupled Case | Same sweep under the decoupled regime. |
| 5b | Chirp Mass Sweep | Keeps distance uncertainty fixed, sweeps chirp mass to probe how binary evolution affects pulsar-term contributions. |

Additional sections cover **numerical stability assessment** (antenna-pattern singularity protection, arccos clamping), **Schur-complement marginalization** details, and **memory profiling / scalability** strategies for >100 pulsars.

---

## Project Structure

### Core Modules

| Module | Description |
|--------|-------------|
| `constants.py` | Physical constants (speed of light, parsec, solar mass, etc.) |
| `deterministic.py` | Enterprise-compatible CW and fuzzy-dark-matter deterministic signal blocks |
| `utils.py` | Single-pulsar `CW_Signal` class, `compute_fisher`, waveform helpers |
| `utils_vectorized.py` | First JAX-vectorized rewrite (`PulsarBatch`, `get_delay_batch`, etc.) |
| `utils_vectorized_fixed.py` | **Current production module** — fixes described above |

### Notebooks

| Notebook | Purpose | Status |
|----------|---------|--------|
| `ring_pterm_vectorized_fixed.ipynb` | **Production analysis** — studies listed above, with corrected physics | ✅ Current |
| `ring_pterm_vectorized.ipynb` | Prior vectorized analysis (superseded by _fixed) | Archived |
| `ring_pterm.ipynb` | Early pulsar-term exploration, per-pulsar `CW_Signal` loop | Legacy |
| `ring_pterm_streamlined.ipynb` | Refactored version with HEALPix maps, linked vs decoupled comparison | Legacy |
| `ring_pterm_optimized.ipynb` | Adds JIT compilation to the streamlined notebook | Legacy |
| `ring.ipynb` | Original ring study without pulsar-term effects | Baseline |

#### Validation & Test Notebooks

| Notebook | Purpose |
|----------|---------|
| `tests.ipynb` | Unit tests for utility functions and Fisher calculations |
| `signal_tests.ipynb` | Waveform validation (face-on vs edge-on, signal properties) |

### Output Directories

| Directory | Contents |
|-----------|----------|
| `outputs_fixed/` | Figures from `ring_pterm_vectorized_fixed.ipynb` |
| `outputs/` | Figures from earlier notebook versions |

---

## Environment Setup

1. **Create a conda environment**

   ```bash
   conda create -n jax-calc python=3.11 jax jaxlib -c conda-forge
   conda activate jax-calc
   ```

2. **Enable 64-bit precision**

   ```bash
   export JAX_ENABLE_X64=1
   ```

   Or at the top of a notebook: `jax.config.update("jax_enable_x64", True)`.

3. **Install extra dependencies**

   ```bash
   pip install matplotlib tqdm healpy scipy
   ```

   For `deterministic.py`, the [Enterprise PTA](https://github.com/nanograv/enterprise) package is also required.

4. **Run the notebook**

   Open `ring_pterm_vectorized_fixed.ipynb` in JupyterLab or VS Code and execute all cells.

## Contributing

Issues and pull requests are welcome. Please open an issue describing proposed changes before submitting larger patches.

## License

Distributed under the MIT License. See `LICENSE` for details.
