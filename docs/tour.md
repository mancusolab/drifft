# A random walk through `drifft`

`drifft` uses the discrete-time Wright–Fisher process to model how allele frequencies
change under genetic drift, selection, mutation, and changes in population size.
This walkthrough introduces its main features through one connected calculation:
defining a biological model,
evolving an allele-count distribution, computing expectations, and
differentiating the results. Start here to see how the pieces fit together
before exploring the API reference and detailed examples linked along the way.

We'll follow an allele through 40 generations, compute the probability that it
is absent at the end, and measure how that probability changes with selection.
`drifft` tracks the probability of each possible allele count, so this calculation
describes a distribution of outcomes across populations. Run the code blocks in
order after [installation](index.md#installation).

## Start with an allele-count distribution

For $N_e$ diploid individuals, a full distribution has $2N_e+1$ entries.
Entry `i` is the probability of `i` derived copies. Start with one copy in
64 individuals:

```python
import jax
import jax.numpy as jnp
import drifft

# Enable float64 before preparing geometry; the numerical backends require it.
jax.config.update("jax_enable_x64", True)
# Counts run from 0 to 128; all initial probability is at one derived copy.
initial = jnp.zeros(129).at[1].set(1.0)
```

Keep endpoints in the vector to retain loss and fixation states. Use float64
before constructing numerical backends.

## Specify selection and mutation

A **frequency map** converts parental allele frequency to the probability of
sampling a derived copy in the next generation. Compose selection and mutation
in their biological order:

```python
# Apply maps left to right: selection first, then recurrent mutation.
frequency_map = drifft.Composition(
    drifft.AdditiveSelection(s=0.02),
    drifft.Mutation(forward_rate=1e-5, backward_rate=1e-5),
)
```

Positive `s` penalizes the derived allele: genotype fitnesses are
$(1,1-s/2,1-s)$. [`Composition`](api/frequency_maps.md#drifft.Composition) applies maps left to right, so mutation follows
selection here. The [selection and mutation API](api/frequency_maps.md)
covers other fitness models and custom maps.

## Apply one generation or a population history

A **backend** represents the one-generation binomial transition. [`Transition`](api/transitions.md#drifft.Transition)
binds it to the frequency map:

```python
# Prepare size-dependent geometry separately from selection and mutation.
backend = drifft.FFT(pop_size=64)
transition = drifft.Transition(backend, frequency_map=frequency_map)
after_one_generation = transition.advance(initial)
```

For multiple generations, prepare the history once with [`Demography.prepare`](api/demography.md#drifft.Demography.prepare),
then bind its geometry to a frequency map with [`DemographicTransition`](api/demography.md#drifft.DemographicTransition). An
**evolution method** applies the backends across each interval. Start with FFT
and Sequential:

```python
# Hold size 64 for 39 generations, then change to size 32 on generation 40.
history = drifft.Demography(pop_sizes=[64, 32], generations=[40])
geometry = history.prepare(backend_factory=drifft.FFT)
model = drifft.DemographicTransition(
    geometry,
    frequency_map=frequency_map,
    method=drifft.Sequential(),
)
# Start from the original distribution, not the one-generation result above.
result = model.advance(initial)
final = result.distribution
```

This evolves `initial` for 39 generations at 64 diploid individuals, then
changes to 32 on generation 40, independently of the one-generation calculation.
Backends and [`Demography`](api/demography.md#drifft.Demography) both use **diploid sizes**; allele counts run from zero
to twice that size.
The [bottleneck example](examples/bottleneck.md) explains boundary timing.

## Compute a statistic

The probability of zero derived copies at generation 40 is:

```python
# Count zero gives absence at this time; recurrent mutation permits later re-entry.
absent_probability = final[0]
print(f"Probability of zero copies: {absent_probability:.6f}")
```

With recurrent mutation, an allele can leave and re-enter the zero-copy state.
This statistic is the probability of absence at the final time, not of having
hit zero at any earlier time.

To evaluate an observable for every possible initial count, use `expect()`:

```python
# Reward absence with one and all other terminal counts with zero.
absent_indicator = jnp.zeros(2 * history.final_pop_size + 1).at[0].set(1.0)
# Propagate the reward backward to obtain absence probabilities for each start count.
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

Keep geometry outside the differentiated function. Construct the frequency map
and bind a lightweight transition inside it, using the current parameters:

```python
def probability_absent(s):
    # Selection depends on s; mutation rates and demographic geometry stay fixed.
    process = drifft.Composition(
        drifft.AdditiveSelection(s=s),
        drifft.Mutation(forward_rate=1e-5, backward_rate=1e-5),
    )
    transition = drifft.DemographicTransition(geometry, process)
    return transition.advance(initial).distribution[0]


# Differentiate the scalar absence probability with respect to s.
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
