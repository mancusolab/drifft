# Population histories

`Demography` describes population sizes and timing. `DemographicTransition`
combines that history with a numerical backend, an epoch method, and a shared
frequency map. It evaluates allele-count distributions deterministically. For a
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
    backend=drifft.FFT,
    frequency_map=drifft.Underdominance(0.01),
)
initial = jnp.zeros(2 * history.initial_ne + 1).at[1].set(1.0)
result = model.advance(initial)
final = result.distribution
```

## Timing and units

Sizes run oldest to newest and are positive integer **diploid** effective
population sizes. The allele-count grid has `2 * ne + 1` entries. Lists or
concrete one-dimensional integer arrays are accepted; floating-point sizes are
not silently rounded.

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

With the default Sequential method, `model.expect(values)` maps final-state
values to expectations conditional on each initial allele count. Its input has
`2 * history.final_ne + 1` entries and
its output has `2 * history.initial_ne + 1` entries. It is the adjoint of
the distribution returned by `.advance()`, not backward population evolution.
Both methods accept signed vectors.

The frequency map applies on the source grid at every generation, including
size changes. Update an Equinox selection model without rebuilding geometry:

```python
terminal_frequency = jnp.linspace(0.0, 1.0, 2 * history.final_ne + 1)

def expected_frequency(strength):
    updated = eqx.tree_at(lambda m: m.frequency_map.strength, model, strength)
    return updated.advance(initial).distribution @ terminal_frequency

gradient = jax.grad(expected_frequency)(jnp.asarray(0.01))
```

Construct histories and prepared models outside JIT. Numerical methods compile
automatically. Sizes and durations are fixed, not differentiable parameters;
changing them can trigger recompilation. Gradient memory use can grow with the
number of generations.

## Backend configuration and limits

One backend is prepared per distinct size within each model. Use `drifft.NUFFT`
instead of `drifft.FFT`, or supply a factory with custom settings:

```python
from functools import partial

spectral_model = drifft.DemographicTransition(
    history,
    backend=partial(drifft.NUFFT, accuracy=1e-11, derivative_accuracy=True),
    frequency_map=drifft.Underdominance(0.01),
)
```

`backend` prepares constant-size geometry; `boundary_backend` prepares the
one-generation size-changing transitions. The boundary factory defaults to
`backend` for all three backends.
The method defaults to `Sequential()`. Factories receive **gene-copy counts**,
even though the history uses diploid Ne.

For other backend and method combinations, see
[choosing numerical algorithms](backends.md). The
[Schiffels–Durbin example](../examples/schiffels_durbin.md) uses Jacobi with
RationalKrylov and an explicit FFT boundary backend. `model.expect()` requires
Sequential. For custom orchestration, see [direct epoch operations](../methods/epochs.md).

## Segregating mass and mutation input

`model.advance_segregating(initial, mutation_rate=1e-8)` discards endpoint mass
each generation and adds singleton input equal to the destination gene-copy
count times the mutation rate. Its input and output contain interior counts
only (`2 * ne - 1` entries). This source is distinct from recurrent mutation in
a frequency map. With recurrent mutation, use full-state `advance()` when
endpoint mass should be able to return to the interior. Equilibrium
initialization is supplied by the caller.

## Results and failures

The current API uses one frequency map throughout the history. Forward methods
return a `DemographicResult` containing `distribution`, `converged`, `status`,
and per-interval `diagnostics`. No intermediate distributions or solver
factorizations are retained in the result. With `throw=True` (the default), a
numerical failure raises; with `throw=False`, inspect the result before using
the distribution. Domain errors still raise. A failing member under `vmap`
may fail the whole batch when throwing is enabled.

Before using a result in an analysis, compare its final statistic and gradient
under tighter settings. The [accuracy guide](numerics.md) explains what interval
diagnostics do and do not establish.
