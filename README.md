# drifft

drifft is a JAX-based Python library for Wright–Fisher population genetics. It
propagates allele-count distributions, computes expectations and expected visits
before absorption, and supports differentiation through selection parameters.
To speed up calculations, it leverages Fast Fourier Transform (FFT) approaches,
hence the name. However it has expanded beyond only FFT-based approaches to
encompass more general compression techniques. It is end-to-end differentiable
using JAX.

[Read the usage and numerical guide](docs/index.md)

## Installation

drifft requires Python 3.12 or newer. Install from a repository checkout with uv:

```bash
git clone https://github.com/mancusolab/drifft.git
cd drifft
uv sync
```

The `jax-finufft` dependency currently builds from Git and requires a C++ toolchain.
See [dependency and hardware constraints](docs/guide/installation.md#dependencies-and-hardware)
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

# Start with one mutant copy and reuse the compiled operation across generations.
distribution = jnp.zeros(2001).at[1].set(1.0)
next_distribution = transition.advance(distribution)
```

`next_distribution[i]` is the probability of `i` mutant copies in the next
generation. Reuse the backend across selection strengths, or substitute
`drifft.NUFFT` to use the Chebyshev/NUFFT implementation.

The guide covers [custom selection and the API](docs/guide/transitions.md),
[demographic histories](docs/guide/demography.md),
[constant-size epochs and immigration](docs/guide/epochs.md),
[backend tradeoffs](docs/guide/backends.md),
[accuracy settings](docs/guide/numerics.md), and
[expected visits and solver limitations](docs/guide/solves.md).
Always check `result.converged` when using `transition.expected_visits()`.

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest -p no:capture
uv run pre-commit run --all-files
```

See [validation and benchmarks](docs/benchmarks.md) for
benchmark commands and [epoch comparisons](benchmarks/EPOCHS.md) for results
against fastDTWF. The [demographic timing report](benchmarks/RATIONAL_KRYLOV.md)
compares Sequential FFT with Jacobi + RationalKrylov on a human population history.

Preview the documentation with `uv run --group docs zensical serve`; see
[documentation development](docs/contributing.md#documentation) for build instructions.

## Support

Report bugs and feature requests through the
[GitHub issue tracker](https://github.com/mancusolab/drifft/issues).
