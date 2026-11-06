# `drifft`

`drifft` is a JAX-based Python library for Wright–Fisher population genetics. It
propagates allele-count distributions, computes expectations and expected visits
before absorption, and supports differentiation through selection parameters.
FFT and polynomial compression accelerate the discrete-time binomial transition.

[Read the usage and numerical guide](https://mancusolab.github.io/drifft/)

## Installation

`drifft` requires Python 3.12 or newer. Install from a repository checkout with uv:

```bash
git clone https://github.com/mancusolab/drifft.git
cd drifft
uv sync
```

The `jax-finufft` dependency currently builds from Git and requires a C++ toolchain.
See [dependency and hardware constraints](https://mancusolab.github.io/drifft/#dependencies-and-hardware)
for build details and device support.

## Quick example

Run this Python example with `uv run python` to advance an allele-count
distribution by one generation:

```python
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)

# 2,000 gene copies represent 1,000 diploid individuals.
backend = drifft.FFT(pop_size=1000)
transition = drifft.Transition(
    backend, frequency_map=drifft.Underdominance(strength=0.01)
)

# Start with one mutant copy.
distribution = jnp.zeros(transition.source_size()).at[1].set(1.0)
next_distribution = transition.advance(distribution)
```

`next_distribution[i]` is the probability of `i` mutant copies in the next
generation. Reuse the backend across selection strengths, or substitute
[`NUFFT`](https://mancusolab.github.io/drifft/api/backends/#drifft.NUFFT) to use the Chebyshev/NUFFT implementation.

Continue with [population histories](https://mancusolab.github.io/drifft/api/demography/),
[expected visits](https://mancusolab.github.io/drifft/api/transitions/#drifft.Transition.expected_visits), or
[accuracy checks](https://mancusolab.github.io/drifft/guide/numerics/).

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest -p no:capture
uv run pre-commit run --all-files
```

See [contributing](https://mancusolab.github.io/drifft/contributing/) for test conventions and documentation
builds, and [benchmarks](https://mancusolab.github.io/drifft/benchmarks/) for numerical comparisons.

## Support

Report bugs and feature requests through the
[GitHub issue tracker](https://github.com/mancusolab/drifft/issues).
