# Specifying selection and mutation

A frequency map takes a source allele frequency and returns an offspring
allele probability. Choose the biological model first; the transition then
uses its probabilities in binomial reproduction.

```python
import jax
import drifft

jax.config.update("jax_enable_x64", True)
selection = drifft.AdditiveSelection(s_het=0.01)
transition = drifft.Transition(drifft.FFT(64), frequency_map=selection)
```

The derived allele has fitnesses $(1,0.99,0.98)$ in this example. Other built-in
models change those genotype fitnesses while keeping the same frequency-map
interface. The [model guide](model.md#selection-and-mutation) derives the update.

## Choose genotype fitnesses

`DiploidSelection(s_het, s_hom)` provides the general diploid selection rule.
Its two required coefficients set heterozygote and derived-homozygote fitness
reductions, respectively. The named models use these constraints:

| Map | Genotype fitnesses (0/0, 0/1, 1/1) | Parameter domain |
| --- | --- | --- |
| `Neutral()` | $(1,1,1)$ | No parameters |
| `DiploidSelection(s_het, s_hom)` | $(1,1-s_{\mathrm{het}},1-s_{\mathrm{hom}})$ | $s_{\mathrm{het}}\le1$, $s_{\mathrm{hom}}<1$ |
| `DominanceSelection(s, h)` | $(1,1-hs,1-s)$ | $s<1$, $hs\le1$ |
| `AdditiveSelection(s_het)` | $(1,1-s_{\mathrm{het}},1-2s_{\mathrm{het}})$ | $s_{\mathrm{het}}<1/2$ |
| `RecessiveSelection(s)` | $(1,1,1-s)$ | $s<1$ |
| `DominantSelection(s)` | $(1,1-s,1-s)$ | $s<1$ |
| `Underdominance(strength)` | $(1,1-s,1)$ | $0\le s\le1$ |
| `Overdominance(strength)` | $(1,1+s,1)$ | $s\ge0$ |
| `MultiplicativeSelection(r)` | $(1,r,r^2)$ | $r>0$ |

All parameters must be finite. Additive and dominance selection also require
finite derived coefficients: `2 * s_het` and `h * s`, respectively.
Negative fitness reductions represent beneficial effects. Scalar shapes are checked at construction; value ranges
are caller preconditions, including under JIT. There is no automatic clipping.
FFT and NUFFT accept all these maps. For Jacobi's model restrictions, see
[choosing numerical algorithms](backends.md). Parameters remain differentiable;
the [population-history example](demography.md#expectations-and-selection-gradients)
shows how to update them in a prepared model.

```python
selection = drifft.DiploidSelection(s_het=0.01, s_hom=0.02)
# The same fitnesses, with one free parameter:
additive = drifft.AdditiveSelection(s_het=0.01)
transition = drifft.Transition(drifft.FFT(64), frequency_map=additive)
```

## Add mutation

`Mutation(forward_rate, backward_rate)` transforms an allele-1 frequency `p` into
`p * (1 - backward_rate) + (1 - p) * forward_rate`. Rates are per-generation
probabilities of 0 → 1 and 1 → 0 mutation, respectively; both default to zero.

`Composition` applies maps **left to right**. This example applies selection
before mutation:

```python
frequency_map = drifft.Composition(
    drifft.Underdominance(strength=0.01),
    drifft.Mutation(forward_rate=1e-8, backward_rate=1e-8),
)
transition = drifft.Transition(drifft.FFT(64), frequency_map=frequency_map)

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
```

Use an Equinox callable with array fields for trainable parameters. For
already evaluated probabilities, see [explicit transition probabilities](transitions.md#explicit-offspring-probabilities).

See the [frequency-map API](../api/frequency_maps.md) for constructor contracts.
