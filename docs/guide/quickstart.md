# Quickstart

What is the probability that a new deleterious allele is lost in the next
generation, and how does that probability change with selection?

After [installation](installation.md), run this example with `uv run python`:

```python
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)

# 32 diploid individuals carry 64 gene copies.
backend = drifft.FFT(gene_copies=64)
selection = drifft.AdditiveSelection(s_het=0.01)
transition = drifft.Transition(backend, frequency_map=selection)

# Exactly one copy of the derived allele is present initially.
initial = jnp.zeros(65).at[1].set(1.0)
after_one_generation = transition.advance(initial)
print(f"Loss probability: {after_one_generation[0]:.4f}")
```

The three genotype fitnesses are $(1,0.99,0.98)$. The returned array describes
all possible offspring counts: entry zero is loss, entry 64 is fixation, and
entries 1 through 63 represent a segregating allele. The loss probability is
about 0.37: an allele starting from a single copy has a substantial chance of
disappearing through random reproduction in the next generation.

## Differentiate the result

`expect` computes an expected destination quantity for every starting count.
An indicator that is one at count zero and zero elsewhere therefore gives
loss probabilities. We can differentiate the probability starting from one copy:

```python
loss_indicator = jnp.zeros(65).at[0].set(1.0)

def loss_probability(s_het):
    selected = drifft.Transition(
        backend, drifft.AdditiveSelection(s_het=s_het)
    )
    return selected.expect(loss_indicator)[1]

sensitivity = jax.grad(loss_probability)(jnp.asarray(0.01))
print(f"Sensitivity to s_het: {sensitivity:.4f}")
```

The derivative is positive: increasing the fitness penalty makes immediate loss
more likely. It is a derivative with respect to the selection coefficient, not
an uncertainty estimate. The calculation reuses the prepared backend while
selection changes.

## Next step

Use [selection and mutation](frequency_maps.md) to change the biological model,
or [population histories](demography.md) to evolve through several generations
and population sizes. [Computing transitions](transitions.md) explains the
relationship between `advance` and `expect` in more detail.
