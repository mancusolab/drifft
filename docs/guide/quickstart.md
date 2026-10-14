# Quickstart

Advance an allele-count distribution by one generation with a fixed population
size and underdominant selection. Run this example with `uv run python`:

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

`next_distribution[i]` is the probability of `i` mutant copies next generation.
Indices zero and `gene_copies` represent loss and fixation. To simulate another
generation, pass `next_distribution` to the same `advance` function.

`transition.expect(values)` computes expected destination values at each source
state. `advance` is its algebraic transpose; neither operation clips or
renormalizes its output.

Reuse the backend across selection strengths. Substitute `drifft.NUFFT(2000)`
for the Chebyshev/NUFFT implementation; see [backend tradeoffs](backends.md).
For custom selection or population-size changes, see [transitions and selection](transitions.md).
