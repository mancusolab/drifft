# Computing transitions

A `Transition` binds prepared numerical geometry to a frequency map. Use
`advance` for a distribution and `expect` for a quantity measured in the next
generation.

```python
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)
backend = drifft.FFT(64)
transition = drifft.Transition(backend, drifft.AdditiveSelection(s_het=0.01))
initial = jnp.zeros(65).at[1].set(1.0)
next_distribution = transition.advance(initial)
allele_frequency = jnp.arange(65) / 64
conditional_mean = transition.expect(allele_frequency)
mean_from_one_copy = conditional_mean[1]
```

`next_distribution[j]` is the probability of count `j` after reproduction.
`conditional_mean[i]` is the expected offspring frequency starting at count
`i`. Consequently, `initial @ conditional_mean` and
`next_distribution @ allele_frequency` agree up to roundoff. This transpose
relationship uses the same approximate transition in both directions.

## Reusing geometry

Construct backends outside JIT and reuse them when changing selection:

```python
stronger = drifft.Transition(backend, drifft.AdditiveSelection(s_het=0.02))
updated_distribution = stronger.advance(initial)
```

Numerical methods compile automatically on their first call. Changing array
shapes or static configuration can require recompilation. Frequency-map
parameters remain dynamic and differentiable; see
[accuracy and differentiation](numerics.md).

## Restricting to segregating counts

```python
interior_mass = transition.advance_segregating(initial[1:-1])
interior_values = transition.expect_segregating(allele_frequency[1:-1])
```

Interior vectors exclude counts zero and `gene_copies`. Their total mass may
decrease because the transition discards trajectories that hit an endpoint.
Outputs are not clipped or renormalized.

!!! note "Discarding is not conditioning"

    Segregating propagation retains the probability of still segregating.
    It does not divide by that probability to condition on survival. With
    recurrent mutation, discarded trajectories do not later re-enter.

For occupation quantities through absorption, see [expected visits](solves.md).

## Explicit offspring probabilities

```python
explicit = drifft.Transition.from_probabilities(backend, transition.probabilities)
```

This binds already evaluated probabilities. To differentiate through their
construction, construct them inside the differentiated function. For built-in
or custom biological models, use a [frequency map](frequency_maps.md).

For a sequence of population-size changes, continue with
[population histories](demography.md). The
[transition API](../api/transitions.md) describes source and destination grids,
input contracts, and the Lineax operator view.
