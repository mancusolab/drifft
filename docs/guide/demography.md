# Population histories

`Demography` describes population sizes and timing. `DemographicTransition`
combines that history with a numerical backend, an evolution method, and one
frequency map used throughout the history. Supply an initial distribution to
compute the distribution at the final population size. For a
complete biological comparison, see the [bottleneck example](../examples/bottleneck.md).

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)

history = drifft.Demography(
    ne=[100, 60, 120],
    generations=[20, 10],
)
model = drifft.DemographicTransition(
    history,
    backend_factory=drifft.FFT,
    frequency_map=drifft.Underdominance(0.01),
)
initial = jnp.zeros(2 * history.initial_ne + 1).at[1].set(1.0)
result = model.advance(initial)
final = result.distribution
```

## Timing and units

Sizes run oldest to newest and are positive integer **diploid** effective
population sizes. The allele-count grid has `2 * ne + 1` entries. Lists or
concrete one-dimensional integer arrays are accepted.

There must be one fewer duration than sizes. Each positive integer duration
counts the generations **between population boundaries**: hold the source size
for `duration - 1` generations, then change to the next size on the final
generation. The example above evaluates:

| Generations | Operation |
| --- | --- |
| 1–19 | Evolve at Ne = 100 |
| 20 | Transition from Ne = 100 to Ne = 60 |
| 21–29 | Evolve at Ne = 60 |
| 30 | Transition from Ne = 60 to Ne = 120 |

!!! note "Boundary timing"

    The size-changing step applies the shared frequency map and binomial
    reproduction. There is no additional resampling step and no subsequent
    evolution at the final size.
    The initial distribution is supplied by the caller; no equilibrium or burn-in
    is assumed.

Repeated sizes are valid: `Demography([100, 100], [50])` evaluates 50 generations
at Ne = 100. `Demography([100], [])` is an identity history. Use
`history.initial_ne`, `history.final_ne`, and `history.total_generations` to
inspect the schedule.

## Expectations and selection gradients

With the default Sequential method, `model.expect(values)` returns a
`DemographicExpectation`. Its `values` field contains expectations conditional
on each initial allele count; `offset` contains the reward from singleton input.
The input vector has `2 * history.final_ne + 1` entries, and `result.values`
has `2 * history.initial_ne + 1` entries. Without singleton input,
`initial @ model.expect(values).values == model.advance(initial).distribution @ values`
up to numerical roundoff.
Both methods accept signed vectors.

Use `model.expect(values, segregating=True)` for interior states, with
`2 * history.final_ne - 1` input entries and `2 * history.initial_ne - 1`
entries in `result.values`. Endpoint mass is discarded at every generation,
just as in segregating forward evolution. Recurrent mutation in the frequency map is
supported. Expectations require `Sequential`; adaptive Krylov approximations
do not provide this matched transpose.

The frequency map applies on the source grid at every generation, including
size changes. Update an Equinox selection model without rebuilding geometry:

```python
terminal_frequency = jnp.linspace(0.0, 1.0, 2 * history.final_ne + 1)

def expected_frequency(strength):
    updated = eqx.tree_at(lambda m: m.frequency_map.strength, model, strength)
    return updated.advance(initial).distribution @ terminal_frequency

gradient = jax.grad(expected_frequency)(jnp.asarray(0.01))
```

Construct the history and model outside the differentiated function; update
only biological parameters inside it. See [differentiating biological parameters](numerics.md#differentiating-biological-parameters)
for supported inputs and gradient checks.

## Backend configuration and limits

A history needs transition geometry for several population sizes, so pass a
backend constructor or factory rather than a single backend instance. To
customize its settings, supply a factory such as:

```python
from functools import partial

spectral_model = drifft.DemographicTransition(
    history,
    backend_factory=partial(drifft.NUFFT, accuracy=1e-11, derivative_accuracy=True),
    frequency_map=drifft.Underdominance(0.01),
)
```

`backend_factory` prepares constant-size geometry;
`boundary_backend_factory` prepares the one-generation size-changing
transitions. The boundary factory defaults to
`backend_factory` for every backend.
The method defaults to `Sequential()`. Factories receive **gene-copy counts**,
even though the history uses diploid Ne.

For other backend and method combinations, see
[choosing numerical algorithms](backends.md). The
[Schiffels–Durbin example](../examples/schiffels_durbin.md) uses Jacobi with
RationalKrylov and an explicit FFT boundary backend. For custom orchestration,
see [direct epoch operations](epochs.md).

## Segregating mass and mutation input

Singleton input represents new mutations entering the segregating population
each generation. It is a source of new mass, distinct from recurrent mutation
of alleles already represented in the distribution.

`model.advance(interior_initial, segregating=True, singleton_mutation_rate=1e-8)` adds singleton mass
equal to the destination gene-copy count times the mutation rate after each
generation. Input and output contain interior counts only (`2 * ne - 1`
entries), with [endpoint mass discarded](transitions.md#restricting-to-segregating-counts).
Use full-state `advance()` when recurrent mutation should allow endpoint re-entry.
For arbitrary source vectors, see [direct epoch operations](epochs.md#mass-and-immigration).

The result includes the reward contributed by singleton input as a scalar offset:

```python
interior_initial = initial[1:-1]
values = jnp.ones(2 * history.final_ne - 1)  # count terminal segregating mass
expectation = model.expect(
    values,
    segregating=True,
    singleton_mutation_rate=1e-8,
)
reward = interior_initial @ expectation.values + expectation.offset
forward = model.advance(
    interior_initial, segregating=True, singleton_mutation_rate=1e-8
)
assert jnp.allclose(reward, forward.distribution @ values)
```

The expectation vector accounts for initial mass; the scalar offset accounts
for newly injected mass. Slicing `initial[1:-1]` discards any initial endpoint
mass without renormalizing it.
Without a source, or for an identity history, `offset` is zero. Both linear and
affine expectations return the same `DemographicExpectation` type.
The expectation operation uses the same Sequential restriction with or without
singleton input.

## Results and failures

Forward methods return the final distribution together with convergence
information. Check that status before using a result from a call that disables
exceptions.

`DemographicExpectation` contains only `values` and `offset`, without a
convergence status. As with `Transition.expect`, finite input values are a
caller precondition. Singleton rates are checked even for identity histories.

The `DemographicResult` contains `distribution`, `converged`, `status`, and
per-interval `diagnostics`. It does not retain intermediate distributions.
Numerical failure raises by default; with `throw=False`, inspect `converged`
and `status`. Domain errors still raise, and a failing member under `vmap`
may fail the batch when throwing is enabled.

The [accuracy guide](numerics.md) explains how to validate the final statistic
and gradient beyond checking convergence.
