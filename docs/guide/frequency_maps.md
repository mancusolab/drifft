# Specifying selection and mutation

A frequency map turns a source allele frequency into an offspring allele
probability. Choose the biological model, then give that map to a transition:

```python
import jax
import drifft

jax.config.update("jax_enable_x64", True)
selection = drifft.AdditiveSelection(s_het=0.01)
transition = drifft.Transition(drifft.FFT(64), frequency_map=selection)
```

The genotype fitnesses are $(1,0.99,0.98)$. `AdditiveSelection` assigns twice
the heterozygote penalty to the derived homozygote.

## Choose genotype fitnesses

Use `DiploidSelection` when the heterozygote and derived-homozygote penalties
are independent:

```python
selection = drifft.DiploidSelection(s_het=0.01, s_hom=0.03)
```

These coefficients give fitnesses $(1,0.99,0.97)$. Negative reductions represent
fitness advantages. `Neutral()` gives equal fitness to all three genotypes.
The [selection API](../api/frequency_maps.md#selection) lists the other constrained
models, their fitnesses, and parameter domains.

## Add mutation

`Mutation` specifies per-generation probabilities of 0 → 1 and 1 → 0 mutation.
`Composition` applies maps **left to right**. This applies selection before
mutation:

```python
frequency_map = drifft.Composition(
    selection,
    drifft.Mutation(forward_rate=1e-8, backward_rate=1e-8),
)
transition = drifft.Transition(drifft.FFT(64), frequency_map=frequency_map)
```

Reverse the order to apply mutation before selection. The two orders generally
produce different offspring probabilities. Selection and mutation parameters
remain differentiable.

!!! warning "Keep endpoint mass for recurrent mutation"

    Full-state `advance()` lets mutation move mass away from loss or fixation.
    Segregating methods discard trajectories at their first endpoint hit;
    `expected_visits()` also stops counting there. Discarded trajectories cannot
    re-enter through later mutation.

## Custom frequency maps

Wrap a pure JAX-compatible function that preserves shape and returns
probabilities in `[0, 1]`:

```python
custom_map = drifft.FrequencyMap(lambda frequency: frequency**2)
custom = drifft.Transition(drifft.FFT(64), frequency_map=custom_map)
```

For trainable parameters, use an Equinox callable with array fields, or implement
`AbstractFrequencyMap.apply`. The [frequency-map API](../api/frequency_maps.md)
describes the interface and constructor contracts.
