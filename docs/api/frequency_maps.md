# Frequency maps

Frequency maps determine offspring allele probabilities. Selection, mutation,
and composition share this interface; custom maps implement the same contract.
See [selection and mutation](../guide/transitions.md#selection-and-mutation)
for usage and parameter domains.

## Interface and custom maps

Implement `AbstractFrequencyMap` or wrap a pure JAX-compatible callable with
`FrequencyMap`. Each map preserves array shape and returns valid offspring
probabilities. See [custom frequency maps](../guide/transitions.md) for usage.

::: drifft.AbstractFrequencyMap
    options:
        members: [apply, __call__]

::: drifft.FrequencyMap
    options:
        members: [__init__]

## Selection

Selection maps genotype fitnesses to offspring allele probabilities under
random mating. All named models below are configurations of the diploid rule;
see [selection models](../guide/transitions.md#selection-and-mutation) for domains
and [the probability model](../guide/model.md#selection-and-mutation) for derivations.

| Model | Fitnesses (0/0, 0/1, 1/1) |
| --- | --- |
| `Neutral` | $(1,1,1)$ |
| `DiploidSelection` | $(1,1-s_{\mathrm{het}},1-s_{\mathrm{hom}})$ |
| `DominanceSelection` | $(1,1-hs,1-s)$ |
| `AdditiveSelection` | $(1,1-s_{\mathrm{het}},1-2s_{\mathrm{het}})$ |
| `RecessiveSelection` | $(1,1,1-s)$ |
| `DominantSelection` | $(1,1-s,1-s)$ |
| `Underdominance` | $(1,1-s,1)$ |
| `Overdominance` | $(1,1+s,1)$ |
| `MultiplicativeSelection` | $(1,r,r^2)$ |

::: drifft.DiploidSelection
    options:
        members: [__init__]

::: drifft.DominanceSelection
    options:
        members: [__init__]

::: drifft.AdditiveSelection
    options:
        members: [__init__]

::: drifft.RecessiveSelection
    options:
        members: [__init__]

::: drifft.DominantSelection
    options:
        members: [__init__]

::: drifft.Underdominance
    options:
        members: [__init__]

::: drifft.Overdominance
    options:
        members: [__init__]

::: drifft.MultiplicativeSelection
    options:
        members: [__init__]

::: drifft.Neutral
    options:
        members: [__init__]

## Mutation and composition

`Mutation` applies forward and back mutation. `Composition` applies frequency
maps in the supplied left-to-right order. See
[selection and mutation](../guide/transitions.md#selection-and-mutation) for
examples and the consequences of changing their order.

::: drifft.Mutation
    options:
        members: [__init__]

::: drifft.Composition
    options:
        members: [__init__]
