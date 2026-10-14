# Installation

drifft requires Python 3.12 or newer. From a repository checkout:

```bash
git clone https://github.com/mancusolab/drifft.git
cd drifft
uv sync
```

Run Python scripts with `uv run python`. Both backends require float64;
enable `jax_enable_x64` before constructing a backend, as shown in the
[quickstart](quickstart.md).

## Dependencies and hardware


`jax-finufft` remains a core dependency for the NUFFT implementation. The dependency follows upstream `jax-finufft` `main`; `uv.lock` records the exact
commit. This includes batching fixes absent from release 1.3.1 and permits the
latest JAX on supported Python versions. Installing from Git builds the native
extension and requires a C++ toolchain. The build uses installed FFTW development
libraries when available; otherwise it downloads and builds FFTW automatically
(OpenMP is optional). Ubuntu CI installs `libfftw3-dev` to avoid that extra build.
CPU execution is tested on Apple Silicon. CUDA requires a
compatible CUDA-enabled FINUFFT installation and has not been validated here;
TPU is unsupported. No dense fallback is used when a device lacks NUFFT support.
