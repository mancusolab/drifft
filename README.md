# drifft

drifft is a JAX-based Python library for Wright–Fisher population genetics. It
propagates allele-count distributions, computes expectations and expected visits
before absorption, and supports differentiation through selection parameters.
FFT and polynomial compression accelerate the discrete-time binomial transition.

[Read the usage and numerical guide](docs/index.md)

## Installation

drifft requires Python 3.12 or newer. Install from a repository checkout with uv:

```bash
git clone https://github.com/mancusolab/drifft.git
cd drifft
uv sync
```

The `jax-finufft` dependency currently builds from Git and requires a C++ toolchain.
See [dependency and hardware constraints](docs/index.md#dependencies-and-hardware)
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
backend = drifft.FFT(gene_copies=2000)
transition = drifft.Transition(
    backend, frequency_map=drifft.Underdominance(strength=0.01)
)

# Start with one mutant copy.
distribution = jnp.zeros(2001).at[1].set(1.0)
next_distribution = transition.advance(distribution)
```

`next_distribution[i]` is the probability of `i` mutant copies in the next
generation. Reuse the backend across selection strengths, or substitute
`drifft.NUFFT` to use the Chebyshev/NUFFT implementation.

Continue with [population histories](docs/api/demography.md),
[expected visits](docs/api/transitions.md#drifft.Transition.expected_visits), or
[accuracy checks](docs/guide/numerics.md).

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest -p no:capture
uv run pre-commit run --all-files
```

See [contributing](docs/contributing.md) for test conventions and documentation
builds, and [benchmarks](docs/benchmarks.md) for numerical comparisons.

## Support

Report bugs and feature requests through the
[GitHub issue tracker](https://github.com/mancusolab/drifft/issues).
