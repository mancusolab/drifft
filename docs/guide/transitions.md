# Computing transitions

Use `Transition` to propagate a distribution by one generation or compute
an expected offspring quantity conditional on the parental count. It combines
a numerical backend with the [selection and mutation map](frequency_maps.md).
The example uses 64 gene copies (32 diploid individuals), starting from one copy.

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

The backend precomputes grids and weights for fixed population sizes and
numerical settings. Construct it outside JIT, then reuse it while changing
selection or mutation parameters:

```python
stronger = drifft.Transition(backend, drifft.AdditiveSelection(s_het=0.02))
updated_distribution = stronger.advance(initial)
```

Numerical methods compile automatically on their first call. Changing array
shapes or static configuration can require recompilation. Frequency-map
parameters remain dynamic and differentiable; see
[accuracy and differentiation](numerics.md).

## Restricting to segregating counts

Use the segregating operations to retain only trajectories that have not hit
loss or fixation. Pass vectors containing interior counts only:

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

If another calculation already supplies the offspring sampling probabilities,
bind those values directly instead of constructing a frequency map.

```python
explicit = drifft.Transition.from_probabilities(backend, transition.probabilities)
```

To differentiate through the probability calculation, construct the probabilities
inside the differentiated function.

For a sequence of population-size changes, continue with
[population histories](demography.md). The
[transition API](../api/transitions.md) describes source and destination grids,
input contracts, and the Lineax operator view.
