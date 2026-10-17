# Installation

drifft requires Python 3.12 or newer. From a repository checkout:

```bash
git clone https://github.com/mancusolab/drifft.git
cd drifft
uv sync
```

Run Python scripts with `uv run python`. All backends require float64;
enable `jax_enable_x64` before constructing a backend, as shown in the
[quickstart](quickstart.md).

## Dependencies and hardware

`jax-finufft` is a core dependency for NUFFT. The project uses its Git version,
with the exact commit recorded in `uv.lock`. Installing from Git builds the native
extension and requires a C++ toolchain. The build uses installed FFTW development
libraries when available; otherwise it downloads and builds FFTW automatically
(OpenMP is optional). Ubuntu CI installs `libfftw3-dev` to avoid that extra build.
CPU execution is tested on Apple Silicon. CUDA requires a
compatible CUDA-enabled FINUFFT installation and has not been validated here;
TPU is unsupported. No dense fallback is used when a device lacks NUFFT support.
