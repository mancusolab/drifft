# drifft

drifft is a JAX-based Python library for Wright–Fisher population genetics. It
propagates allele-count distributions, computes expectations and expected visits
before absorption, and supports differentiation through selection parameters.
FFT and Chebyshev/NUFFT backends share the same Equinox interface.

[Read the usage and numerical guide](docs/guide.md)

## Installation

drifft requires Python 3.12 or newer. Install from a repository checkout with uv:

```bash
git clone https://github.com/mancusolab/drifft.git
cd drifft
uv sync
```

The `jax-finufft` dependency currently builds from Git and requires a C++ toolchain.
See [dependency and hardware constraints](docs/guide.md#dependency-and-hardware-constraints)
for build details and device support.

## Quick example

Run this Python example with `uv run python` to advance an allele-count
distribution by one generation:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)

# 2,000 gene copies represent 1,000 diploid individuals.
backend = drifft.FFT(gene_copies=2000)
transition = drifft.Transition(
    backend, selection=drifft.Underdominance(strength=0.01)
)

# Start with one mutant copy and reuse the compiled operation across generations.
advance = eqx.filter_jit(drifft.Transition.advance)
distribution = jnp.zeros(2001).at[1].set(1.0)
next_distribution = advance(transition, distribution)
```

`next_distribution[i]` is the probability of `i` mutant copies in the next
generation. Reuse the backend across selection strengths, or substitute
`drifft.NUFFT` to use the Chebyshev/NUFFT implementation.

The guide covers [custom selection and the API](docs/guide.md#installation-and-use),
[backend tradeoffs](docs/guide.md#backend-selection),
[accuracy settings](docs/guide.md#nufft-approximation-and-configuration), and
[expected visits and solver limitations](docs/guide.md#solves-and-limitations).
Always check `result.converged` when using `transition.expected_visits()`.

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest -p no:capture
uv run pre-commit run --all-files
```

See [validation and benchmarks](docs/guide.md#validation-and-benchmarks) for
benchmark commands and [epoch comparisons](benchmarks/EPOCHS.md) for results
against fastDTWF.

## Support

Report bugs and feature requests through the
[GitHub issue tracker](https://github.com/mancusolab/drifft/issues).
