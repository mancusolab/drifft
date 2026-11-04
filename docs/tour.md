# A random walk through drifft

Compute the probability that a derived allele is absent after 40 generations,
then differentiate that probability with respect to selection. This tour
propagates the full allele-count distribution deterministically; it does not
simulate individual trajectories. Run the code blocks in order after
[installation](index.md#installation).

## Start with an allele-count distribution

For $N_e$ diploid individuals, a full distribution has $2N_e+1$ entries.
Entry `i` is the probability of `i` derived copies. Start with one copy in
64 individuals:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)
initial = jnp.zeros(129).at[1].set(1.0)
```

Keep endpoints in the vector to retain loss and fixation states. Use float64
before constructing numerical backends.

## Specify selection and mutation

A **frequency map** converts parental allele frequency to the probability of
sampling a derived copy in the next generation. Compose selection and mutation
in their biological order:

```python
frequency_map = drifft.Composition(
    drifft.AdditiveSelection(s=0.02),
    drifft.Mutation(forward_rate=1e-5, backward_rate=1e-5),
)
```

Positive `s` penalizes the derived allele: genotype fitnesses are
$(1,1-s/2,1-s)$. `Composition` applies maps left to right, so mutation follows
selection here. The [selection and mutation API](api/frequency_maps.md)
covers other fitness models and custom maps.

## Apply one generation or a population history

A **backend** represents the one-generation binomial transition. `Transition`
binds it to the frequency map:

```python
backend = drifft.FFT(diploid_size=64)
transition = drifft.Transition(backend, frequency_map=frequency_map)
after_one_generation = transition.advance(initial)
```

For multiple generations, use `DemographicTransition`. It prepares backends for
the sizes in the history; an **evolution method** applies them across each
interval. Start with FFT and Sequential:

```python
history = drifft.Demography(ne=[64, 32], generations=[40])
model = drifft.DemographicTransition(
    history,
    frequency_map=frequency_map,
    backend_factory=drifft.FFT,
    method=drifft.Sequential(),
)
result = model.advance(initial)
final = result.distribution
```

This evolves `initial` for 39 generations at 64 diploid individuals, then
changes to 32 on generation 40, independently of the one-generation calculation.
Backends and `Demography` both use **diploid sizes**; allele counts run from zero
to twice that size.
The [bottleneck example](examples/bottleneck.md) explains boundary timing.

## Compute a statistic

The probability of zero derived copies at generation 40 is:

```python
absent_probability = final[0]
print(f"Probability of zero copies: {absent_probability:.6f}")
```

With recurrent mutation, an allele can leave and re-enter the zero-copy state.
This statistic is the probability of absence at the final time, not of having
hit zero at any earlier time.

To evaluate an observable for every possible initial count, use `expect()`:

```python
absent_indicator = jnp.zeros(2 * history.final_ne + 1).at[0].set(1.0)
expectation = model.expect(absent_indicator)
assert jnp.allclose(
    initial @ expectation.values, absent_probability, rtol=0, atol=1e-10
)
```

`advance()` evolves one initial distribution; `expect()` returns conditional
expectations indexed by initial count. Demographic expectations currently
require Sequential. For trajectories stopped at loss or fixation, see
[segregating transitions and expected visits](api/transitions.md#drifft.Transition.advance).

## Differentiate selection

Keep the prepared model outside the differentiated function. `eqx.tree_at`
returns a copy with the selected parameter replaced, reusing the numerical
geometry. Here `maps[0]` is the selection component:

```python
def probability_absent(s):
    updated = eqx.tree_at(lambda m: m.frequency_map.maps[0].s, model, s)
    return updated.advance(initial).distribution[0]

probability, sensitivity = jax.value_and_grad(probability_absent)(jnp.asarray(0.02))
print(f"Probability: {probability:.6f}; derivative: {sensitivity:.6f}")
```

Selection, mutation, and initial mass can be differentiable inputs. Population
sizes, generation counts, and numerical dimensions remain fixed. The
[selection-sensitivity example](examples/selection_sensitivity.md) adds
finite-difference checks and a parameter sweep.

## Choose numerical accuracy separately

Changing the backend or evolution method changes the approximation, not the
biological model. Keep FFT and Sequential until their time or memory cost
motivates an alternative; [choosing numerical algorithms](guide/backends.md)
compares the options.

Numerical failure raises by default. Successful execution still needs
[accuracy checks](guide/numerics.md): compare the statistic and its derivative
under tighter settings and against a small reference problem. The
[computational overview](methods/overview.md) explains the sources of error.
