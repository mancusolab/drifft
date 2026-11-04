# drifft

`drifft` computes allele-count distributions, expectations, and occupation
times under the discrete-time Wright–Fisher model with selection, mutation,
and changing population sizes. Numerical approximations accelerate the binomial
transition without replacing it by a diffusion model. JAX supports derivatives
with respect to selection, mutation, and initial mass.

## Installation

drifft requires Python 3.12 or newer. From a repository checkout:

```bash
git clone https://github.com/mancusolab/drifft.git
cd drifft
uv sync
```

Run Python scripts with `uv run python`. All backends require float64;
enable `jax_enable_x64` before constructing a backend, as shown below.

## Quickstart

Compute one-generation loss probability for a new deleterious allele:

```python
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)

# 32 diploid individuals carry 64 gene copies.
backend = drifft.FFT(pop_size=32)
selection = drifft.AdditiveSelection(s=0.02)
transition = drifft.Transition(backend, frequency_map=selection)

# Exactly one copy of the derived allele is present initially.
initial = jnp.zeros(transition.source_size()).at[1].set(1.0)
after_one_generation = transition.advance(initial)
print(f"Loss probability: {after_one_generation[0]:.4f}")
```

`AdditiveSelection(s)` assigns genotype fitnesses $(1,1-s/2,1-s)$:
positive `s` penalizes the derived allele. Output entry `i` is the probability
of `i` copies; entries 0 and 64 are loss and fixation.

Put the same calculation in a function of the selection coefficient and apply
`jax.grad`:

```python
def loss_probability(s):
    selected = drifft.Transition(backend, drifft.AdditiveSelection(s=s))
    return selected.advance(initial)[0]


sensitivity = jax.grad(loss_probability)(jnp.asarray(0.02))
print(f"Sensitivity to s: {sensitivity:.4f}")
```

The derivative is positive: increasing the fitness penalty makes immediate loss
more likely. The calculation reuses the prepared backend while selection changes.

## Next steps

Continue with [A random walk through drifft](tour.md): connect selection,
mutation, population history, and differentiation in one calculation.

## Dependencies and hardware

`jax-finufft` is a core dependency for NUFFT. The project uses its Git version,
with the exact commit recorded in `uv.lock`. Installing from Git builds the native
extension and requires a C++ toolchain. The build uses installed FFTW development
libraries when available; otherwise it downloads and builds FFTW automatically
(OpenMP is optional).
CPU execution is tested on Apple Silicon. CUDA requires a
compatible CUDA-enabled FINUFFT installation and has not been validated here;
TPU is unsupported. No dense fallback is used when a device lacks NUFFT support.
