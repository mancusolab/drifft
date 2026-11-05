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

# Enable float64 before preparing geometry; the numerical backends require it.
jax.config.update("jax_enable_x64", True)
history = drifft.Demography(pop_sizes=[64, 16, 64], generations=[20, 20])
# Prepare size-dependent geometry once; it can be reused for other mutation rates.
geometry = history.prepare(backend_factory=drifft.FFT)
model = drifft.DemographicTransition(
    geometry,
    # Mutation enters as new singleton mass below, not as recurrent allele changes.
    frequency_map=drifft.Neutral(),
    method=drifft.Sequential(),
)
# Interior states are counts 1 through 2N-1; zero mass means no ancestral variants.
initial = jnp.zeros(2 * history.initial_pop_size - 1)
mutation_rate = 1e-8
result = model.advance(
    initial,
    segregating=True,  # Discard variants at loss or fixation without renormalizing.
    # Add 2 * destination population size * rate at count one each generation.
    singleton_mutation_rate=mutation_rate,
)
# This is expected variant mass per site, not a probability distribution.
expected_counts = result.distribution
print(f"Expected segregating mass per site: {expected_counts.sum():.6g}")
print(f"Expected singleton mass per site: {expected_counts[0]:.6g}")
```

The vector contains only counts 1 through `2 * history.final_pop_size - 1`.
Its entries are expected mass, not probabilities summing to one. This is a
population count spectrum; a sample spectrum additionally requires a sampling
projection, as illustrated in the [sample-likelihood example](schiffels_durbin.md).

## Source timing

After each generation, the model adds `2 * dest_pop_size * mutation_rate`
at count one, including after the last generation. At size-changing steps,
the **destination** size sets this scaling. Thus mutations injected on the
last step remain singletons in the output. The
[bottleneck example](bottleneck.md#when-the-size-changes) explains boundary timing.

The initial state is zero, so all output comes from mutations introduced during
these 40 generations; no ancestral equilibrium is assumed. Endpoint hits are
discarded permanently. Use a full-state distribution and a [`Mutation`](../api/frequency_maps.md#drifft.Mutation) frequency
map instead when modeling recurrent mutation and endpoint re-entry.

## Expectations with a source

For terminal values `f`, the expected reward is
`initial @ expectation.values + expectation.offset`. The offset accounts for
mutations introduced after the initial state:

```python
# Give every surviving count reward one to measure total segregating mass.
values = jnp.ones(2 * history.final_pop_size - 1)
# Use the same stopped process and mutation input as in the forward calculation.
expectation = model.expect(
    values,
    segregating=True,
    singleton_mutation_rate=mutation_rate,
)
# The offset includes variants introduced during the history by the singleton source.
expected_total = initial @ expectation.values + expectation.offset
assert jnp.allclose(expected_total, expected_counts.sum(), rtol=1e-10, atol=1e-14)
```

Here the initial contribution is zero, so `expectation.offset` equals the
expected terminal segregating mass. Demographic expectations require Sequential.
For a constant-size interval with an arbitrary source vector, pass `source_mass`
to [`AbstractEpochMethod.advance`](../api/methods.md#drifft.AbstractEpochMethod.advance).
This vector supplies the mass to add each generation directly;
`singleton_mutation_rate` supplies a per-copy rate that the demographic model
scales by the destination population size.
Validate the spectrum under tighter settings using the [accuracy checks](../guide/numerics.md).
