# New mutations through time

Accumulate the expected number of segregating variants from new mutations
through a population history. Begin with no variation, inject singleton mass
each generation, and discard variants when they reach loss or fixation.
This is a source-driven calculation; it does not use recurrent mutation of
an existing allele.

## Calculation

Run the blocks in order after [installation](../index.md#installation).
The mutation rate below is per gene copy per generation per site, so the
output is expected variant mass per site at each population allele count.

```python
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)
history = drifft.Demography(ne=[64, 16, 64], generations=[20, 20])
model = drifft.DemographicTransition(
    history,
    frequency_map=drifft.Neutral(),
    backend_factory=drifft.FFT,
    method=drifft.Sequential(),
)
initial = jnp.zeros(2 * history.initial_ne - 1)
mutation_rate = 1e-8
result = model.advance(
    initial,
    segregating=True,
    singleton_mutation_rate=mutation_rate,
)
expected_counts = result.distribution
print(f"Expected segregating mass per site: {expected_counts.sum():.6g}")
print(f"Expected singleton mass per site: {expected_counts[0]:.6g}")
```

The vector contains only counts 1 through `2 * history.final_ne - 1`.
Its entries are expected mass, not probabilities summing to one. This is a
population count spectrum; a sample spectrum additionally requires a sampling
projection, as illustrated in the [sample-likelihood example](schiffels_durbin.md).

## Source timing

After each generation, the model adds `2 * destination_ne * mutation_rate`
at count one, including after the last generation. At size-changing steps,
the **destination** size sets this scaling. Thus mutations injected on the
last step remain singletons in the output. The
[bottleneck example](bottleneck.md#when-the-size-changes) explains boundary timing.

The initial state is zero, so all output comes from mutations introduced during
these 40 generations; no ancestral equilibrium is assumed. Endpoint hits are
discarded permanently. Use a full-state distribution and a `Mutation` frequency
map instead when modeling recurrent mutation and endpoint re-entry.

## Expectations with a source

For terminal values `f`, the expected reward is
`initial @ expectation.values + expectation.offset`. The offset accounts for
mutations introduced after the initial state:

```python
values = jnp.ones(2 * history.final_ne - 1)
expectation = model.expect(
    values,
    segregating=True,
    singleton_mutation_rate=mutation_rate,
)
expected_total = initial @ expectation.values + expectation.offset
assert jnp.allclose(expected_total, expected_counts.sum(), rtol=1e-10, atol=1e-14)
```

Here the initial contribution is zero, so `expectation.offset` equals the
expected terminal segregating mass. Demographic expectations require Sequential.
For arbitrary immigration vectors, use [`AbstractEpochMethod.advance`](../api/methods.md#drifft.AbstractEpochMethod.advance).
Validate the spectrum under tighter settings using the [accuracy checks](../guide/numerics.md).
