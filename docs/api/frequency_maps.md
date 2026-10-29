# Frequency maps

Frequency maps determine offspring allele probabilities. Selection, mutation,
and composition share this interface; custom maps implement the same contract.
See [selection and mutation](../guide/frequency_maps.md) for a practical example.

## Interface and custom maps

Implement `AbstractFrequencyMap` or wrap a pure JAX-compatible callable with
`FrequencyMap`. Each map preserves array shape and returns valid offspring
probabilities. See [custom frequency maps](../guide/frequency_maps.md#custom-frequency-maps) for usage.

::: drifft.AbstractFrequencyMap
    options:
        members: [apply, __call__]

::: drifft.FrequencyMap
    options:
        members: [__init__]

## Selection

Selection maps genotype fitnesses to offspring allele probabilities under
random mating. All named models below are configurations of the diploid rule;
see [the probability model](../guide/model.md#selection-and-mutation) for derivations.

| Model | Fitnesses (0/0, 0/1, 1/1) | Parameter domain |
| --- | --- | --- |
| `Neutral` | $(1,1,1)$ | No parameters |
| `DiploidSelection` | $(1,1-s_{\mathrm{het}},1-s_{\mathrm{hom}})$ | $s_{\mathrm{het}}\le1$, $s_{\mathrm{hom}}<1$ |
| `DominanceSelection` | $(1,1-hs,1-s)$ | $s<1$, $hs\le1$ |
| `AdditiveSelection` | $(1,1-s_{\mathrm{het}},1-2s_{\mathrm{het}})$ | $s_{\mathrm{het}}<1/2$ |
| `RecessiveSelection` | $(1,1,1-s)$ | $s<1$ |
| `DominantSelection` | $(1,1-s,1-s)$ | $s<1$ |
| `Underdominance` | $(1,1-s,1)$ | $0\le s\le1$ |
| `Overdominance` | $(1,1+s,1)$ | $s\ge0$ |
| `MultiplicativeSelection` | $(1,r,r^2)$ | $r>0$ |

All parameters and derived coefficients (`2 * s_het`, `h * s`) must be finite.
Constructors check scalar shapes; parameter ranges are caller preconditions
when evaluating frequency maps. Values are not clipped. FFT and NUFFT accept
all these models; [Jacobi](backends.md#jacobi) checks additional parameter bounds.

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
[selection and mutation](../guide/frequency_maps.md) for
examples and the consequences of changing their order.

Mutation rates must be finite and in `[0, 1]`; their sum need not be less than
one. Rates default to zero and remain differentiable array leaves. Scalar
shapes are checked at construction; value ranges are caller preconditions for
direct map evaluation. Jacobi also checks the rate domain at runtime.

::: drifft.Mutation
    options:
        members: [__init__]

::: drifft.Composition
    options:
        members: [__init__]
