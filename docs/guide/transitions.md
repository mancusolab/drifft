# Transitions and frequency maps

The [quickstart](quickstart.md) constructs a backend and binds it to a transition.
The [model guide](model.md) explains the binomial kernel, selection and mutation
equations, and the relationship between propagation and expectations.

Build the backend once and reuse it across transitions. `Transition` binds the backend
without rebuilding it. Frequency-map parameters are dynamic array leaves, while
shape/configuration metadata are static. For another selection strength, construct
`Transition(backend, Underdominance(new_strength))` and call its methods directly.
Scalar strengths are normalized to float64 automatically.

`expect`, `advance`, their segregating variants, and `expected_visits` compile
automatically. Call them directly on the transition. JAX compiles on the first
call for each shape/static configuration and reuses the executable when array
values change. For a whole simulation or inference objective, an outer
`eqx.filter_jit` can still combine multiple method calls into one workflow.

| API | Contract |
| --- | --- |
| `FFT(gene_copies, **options)` / `NUFFT(gene_copies, **options)` | Prepare destination geometry outside JIT |
| `AbstractTransitionBackend` | Equinox abstract base; implement linear `expect(probabilities, values)` |
| `Transition(backend, frequency_map=...)` | Bind an `AbstractFrequencyMap` describing selection, mutation, or both |
| `Transition.from_probabilities(backend, probabilities)` | Bind explicit probabilities, one per source row |
| `transition.expect(values)` | Destination values to source expectations: `P @ values` |
| `transition.advance(distribution)` | Source mass to destination mass: `P.T @ distribution` |
| `transition.expect_segregating(values)` | Interior expectations with endpoint values zero |
| `transition.advance_segregating(distribution)` | Interior mass propagation, discarding endpoint hits |
| `transition.as_lineax(segregating=False)` | Expectation operator; `.T` advances mass |
| `transition.expected_visits(**options)` | Square-transition visits from one copy, with convergence diagnostics |

`expect` and `advance` have different directions and are intentionally named
explicitly. Neither changes the input, clips outputs, nor normalizes probability
mass. Segregating operations remove endpoint contributions; they do not condition
on survival. Repeated operations remain JIT-compatible; NUFFTs use FINUFFT’s compiled FFI
rather than Python host callbacks.

## Custom frequency maps

Every map implements `.apply(frequencies)`; calling the map directly is shorthand
for the same operation. Transitions always use `.apply()`. Implement
`AbstractFrequencyMap.apply` in your own Equinox class, or wrap a function:

```python
import jax
import drifft

jax.config.update("jax_enable_x64", True)
backend = drifft.FFT(2000)

# Wrap a JAX-compatible function that returns one probability per frequency.
custom_map = drifft.FrequencyMap(lambda frequency: frequency**2)
custom = drifft.Transition(backend, frequency_map=custom_map)
# An Equinox callable can carry dynamic parameters for inference or sweeps.
# Alternatively, supply already evaluated probabilities explicitly.
offspring_probabilities = custom.probabilities
explicit = drifft.Transition.from_probabilities(backend, offspring_probabilities)
```

Model probabilities are evaluated during application, keeping parameter updates
and differentiation connected. With explicit probabilities, construct them inside
the differentiated function if gradients through their construction are needed.
Use Equinox fields for trainable parameters instead of closing over changing
Python values in fresh lambdas.

## Selection and mutation

`Neutral()` leaves frequencies unchanged. `Underdominance(strength)` applies
symmetric heterozygote disadvantage. `Mutation(forward_rate, backward_rate)`
transforms an allele-1 frequency `p` into
`p * (1 - backward_rate) + (1 - p) * forward_rate`. Rates are per-generation
probabilities of 0 → 1 and 1 → 0 mutation, respectively; both default to zero.

`Composition` applies maps **left to right**. This example applies selection
before mutation:

```python
frequency_map = drifft.Composition(
    drifft.Underdominance(strength=0.01),
    drifft.Mutation(forward_rate=1e-8, backward_rate=1e-8),
)
transition = drifft.Transition(backend, frequency_map=frequency_map)

frequencies = transition.probabilities
# frequency_map.apply(x) and frequency_map(x) are equivalent.
```

Reverse the component order to apply mutation before selection. Selection and
mutation generally do not commute. Mutation parameters are dynamic float64
scalar leaves and support differentiation just like selection strength. Each
rate must be finite and lie in `[0, 1]`; their sum need not be less than one.
Nonscalar parameters raise `ValueError`, while value ranges remain caller
preconditions under transformations.

!!! warning "Mutation and endpoint removal"

    With forward and back mutation, loss and fixation are no longer absorbing under
    `.advance()`. Segregating methods still discard trajectories at their first
    endpoint hit, and `expected_visits()` still ends visits there. Those operations
    do not include later re-entry through mutation or compute a recurrent-mutation
    stationary distribution.

## Population grids and validation

For population-size changes, the backend specifies destination size and
`Transition(backend, frequency_map=..., source_gene_copies=...)` specifies source size.
`expect` maps destination-length values to source-length expectations; `advance`
performs the reverse mapping for mass. Explicit probabilities may also describe
sampled source rows. Segregating and expected-visits operations assume complete,
ordered allele-count grids; expected visits additionally requires equal source
and destination sizes. For a sequence of size changes, use
[`Demography` and `DemographicTransition`](demography.md).

!!! warning "Input validation"

    Inputs must have the documented shapes. Frequencies and probabilities must be in
    `[0, 1]`, and `Underdominance.strength` must be scalar and in `[0, 1]`. Shape checks
    raise during construction or tracing; finite values and numerical ranges remain
    caller preconditions to avoid runtime callbacks. Backends require float64.
