# PTA Continuous-Wave Fisher Analysis

This repository contains tooling for exploring continuous gravitational-wave (CW) signals in pulsar timing array (PTA) data using JAX-backed likelihood evaluations and Fisher-matrix approximations. The code started as a port of ideas from `gabefreedman/etudes` and currently focuses on fast deterministic signal generation, parameter estimation utilities, and companion notebooks for experimentation.

## Project Structure

### Core Modules

- `constants.py` – numerical constants (speed of light, parsec, solar mass, etc.) shared across modules.
- `deterministic.py` – builders for CW and fuzzy-dark-matter deterministic signal blocks that plug into Enterprise PTA model pipelines.
- `utils.py` – assorted helper routines for waveform generation, interpolation, and PTA-specific math.
- `utils_vectorized.py` – JAX-optimized vectorized versions of utility functions with batched operations via `PulsarBatch` class.
- `omega_interpolation_data.npz` – cached interpolation tables consumed by the utilities.
- `outputs/` – checkpointed figures and intermediate results produced by the notebooks.

### Notebooks

The notebooks form a progression from initial exploration to optimized analysis:

#### Primary Analysis Notebooks (Ring Studies)

| Notebook | Purpose | Status |
|----------|---------|--------|
| `ring.ipynb` | Initial exploration of angular resolution vs pulsar proximity to GW source. Uses basic Fisher matrix approach without pulsar-term effects. | Baseline |
| `ring_pterm.ipynb` | Extends `ring.ipynb` to include pulsar-term contributions to the timing residual model. Explores how pulsar distance uncertainty affects sky localization. | Extended |
| `ring_pterm_streamlined.ipynb` | Refactored version with cleaner code structure, HEALPix all-sky maps, and systematic comparison of linked (known phase) vs decoupled (unknown phase) regimes. | Refactored |
| `ring_pterm_optimized.ipynb` | Adds JAX JIT compilation to `ring_pterm_streamlined.ipynb` for faster Fisher matrix calculations while keeping the same analysis structure. | JIT-enabled |
| `ring_pterm_vectorized.ipynb` | **Most advanced version.** Fully vectorized across pulsars using JAX vmap with `PulsarBatch` PyTree. Includes numerical stability improvements, flexible `n_psrs` configuration, and memory profiling for scalability to 100+ pulsars. | Production |

**Recommended workflow**: Start with `ring_pterm_vectorized.ipynb` for production analysis. Refer to earlier notebooks to understand the conceptual progression.

#### Validation & Test Notebooks

| Notebook | Purpose |
|----------|---------|
| `tests.ipynb` | General unit tests for utility functions, Fisher matrix calculations, and CW signal models. |
| `signal_tests.ipynb` | Validates CW signal generation by comparing face-on vs edge-on binary orientations and checking waveform properties. |
| `sprite_tests.ipynb` | Exploratory analysis of SPRITE catalog data (massive black hole binaries), computing expected angular separations at PTA frequencies. |

## Getting Started

1. **Create the environment**

   ```bash
   conda create -n jax-calc python=3.11 jax jaxlib -c conda-forge
   ```

   Activate with `conda activate jax-calc`, then enable 64-bit precision by exporting `JAX_ENABLE_X64=1` or calling `jax.config.update("jax_enable_x64", True)` inside notebooks.

2. **Install dependencies**

   Install Enterprise PTA and any extra packages the notebooks rely on (e.g., `matplotlib`, `tqdm`) either via pip or conda once the environment is active.

3. **Open the notebooks**

   Launch JupyterLab or VS Code and run the notebooks to reproduce the analyses. Many notebooks expect the `omega_interpolation_data.npz` file to stay in place.

## Typical Workflow

1. Assemble a deterministic signal model with `deterministic.cw_block_circ` or `deterministic.cw_block_ecc`.
2. Combine blocks with additional PTA noise models inside Enterprise to build a full timing model.
3. Evaluate Fisher matrices and waveform residuals either programmatically or via the provided notebooks.
4. Export plots or tables to `outputs/` for downstream use.

## Contributing

Issues and pull requests are welcome. Please open an issue describing proposed changes or feature requests before submitting larger patches.

## License

Distributed under the MIT License. See `LICENSE` for details.
