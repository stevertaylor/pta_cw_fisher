# PTA Continuous-Wave Fisher Analysis

This repository contains tooling for exploring continuous gravitational-wave (CW) signals in pulsar timing array (PTA) data using JAX-backed likelihood evaluations and Fisher-matrix approximations. The code started as a port of ideas from `gabefreedman/etudes` and currently focuses on fast deterministic signal generation, parameter estimation utilities, and companion notebooks for experimentation.

## Project Structure

- `constants.py` – numerical constants (speed of light, parsec, solar mass, etc.) shared across modules.
- `deterministic.py` – builders for CW and fuzzy-dark-matter deterministic signal blocks that plug into Enterprise PTA model pipelines.
- `utils.py` – assorted helper routines for waveform generation, interpolation, and PTA-specific math.
- `omega_interpolation_data.npz` – cached interpolation tables consumed by the utilities.
- `ring.ipynb`, `ring_pterm.ipynb`, `signal_tests.ipynb`, `sprite_tests.ipynb`, `tests.ipynb` – exploratory notebooks comparing signal models, probing pulsar-term impacts, and validating Fisher estimates.
- `outputs/` – checkpointed figures and intermediate results produced by the notebooks.

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
