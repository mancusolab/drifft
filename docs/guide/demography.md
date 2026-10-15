# Demographic histories

`Demography` describes population sizes and timing. `DemographicTransition`
combines that history with a numerical backend and a shared frequency map.
It evaluates allele-count distributions deterministically.

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
final = model.advance(initial)
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

    The size-changing step includes selection and binomial reproduction. There is
    no additional resampling step and no subsequent evolution at the final size.
    The initial distribution is supplied by the caller; no equilibrium or burn-in
    is assumed. This timing differs from fastDTWF's destination-size intervals,
    which change size at the start of each interval.

Repeated sizes are valid: `Demography([100, 100], [50])` evaluates 50 generations
at Ne = 100. `Demography([100], [])` is an identity history. Use
`history.initial_ne`, `history.final_ne`, and `history.total_generations` to
inspect the schedule.

## Expectations and selection gradients

`model.expect(values)` maps final-state values to expectations conditional on
each initial allele count. Its input has `2 * history.final_ne + 1` entries and
its output has `2 * history.initial_ne + 1` entries. It is the adjoint of
`.advance()`, not backward population evolution. Both methods accept signed
vectors and leave inputs unchanged; neither clips nor renormalizes outputs.

The frequency map applies on the source grid at every generation, including
size changes. Update an Equinox selection model without rebuilding geometry:

```python
terminal_frequency = jnp.linspace(0.0, 1.0, 2 * history.final_ne + 1)

def expected_frequency(strength):
    updated = eqx.tree_at(lambda m: m.frequency_map.strength, model, strength)
    return updated.advance(initial) @ terminal_frequency

gradient = jax.grad(expected_frequency)(jnp.asarray(0.01))
```

Construct histories and prepared models outside JIT. Both numerical methods
compile automatically. Sizes and durations are fixed metadata, not differentiable
parameters; changing them can trigger recompilation. Constant-size portions use
fixed-length loops supporting reverse-mode differentiation. Shapes change only
between intervals. Reverse-mode memory can grow with the number of generations.

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

The current API uses one frequency map throughout the history and returns
only the final distribution. Use [frequency-map composition](transitions.md#selection-and-mutation)
to combine selection and mutation in an explicit order.

!!! warning "Accumulated error"

    Each generation inherits the chosen backend's
    approximation and derivative limitations; errors can accumulate over long
    histories. See [numerical accuracy](numerics.md). Small dense-reference tests
    validate timing and gradients; they do not establish accuracy or performance
    for every long demographic history.
