# Transitions and selection

The [quickstart](quickstart.md) constructs a backend and binds it to a transition.

Build the backend once and reuse it across transitions. `Transition` binds the backend
without rebuilding it. Selection parameters are dynamic array leaves, while
shape/configuration metadata are static. For another selection strength, construct
`Transition(backend, Underdominance(new_strength))` and pass it to the same compiled
function. Scalar strengths are normalized to float64 automatically.

| API | Contract |
| --- | --- |
| `FFT(gene_copies, **options)` / `NUFFT(gene_copies, **options)` | Prepare destination geometry outside JIT |
| `AbstractTransitionBackend` | Equinox abstract base; implement linear `expect(probabilities, values)` |
| `Transition(backend, selection=...)` | Bind a JAX-compatible frequency-to-probability callable |
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

Custom selection models use the same interface:

```python
import jax
import drifft

jax.config.update("jax_enable_x64", True)
backend = drifft.FFT(2000)

# Any JAX-compatible callable that returns one probability per frequency works.
custom = drifft.Transition(backend, selection=lambda frequency: frequency**2)
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

For population-size changes, the backend specifies destination size and
`Transition(backend, selection=..., source_gene_copies=...)` specifies source size.
`expect` maps destination-length values to source-length expectations; `advance`
performs the reverse mapping for mass. Explicit probabilities may also describe
sampled source rows. Segregating and expected-visits operations assume complete,
ordered allele-count grids; expected visits additionally requires equal source
and destination sizes. A full changing-size demographic workflow is not provided.

Inputs must have the documented shapes. Frequencies and probabilities must be in
`[0, 1]`, and `Underdominance.strength` must be scalar and in `[0, 1]`. Shape checks
raise during construction or tracing; finite values and numerical ranges remain
caller preconditions to avoid runtime callbacks. Backends require float64.
