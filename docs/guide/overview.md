# Overview

`drifft` evolves allele-count distributions under the discrete-time
Wright–Fisher model. A frequency map determines the probability of sampling
an allele in the next generation; a numerical backend approximates the
corresponding binomial transition.

## One-generation transitions

Choose a [frequency map](../api/frequency_maps.md) to describe selection, mutation,
or their composition. Prepare an [FFT](../api/backends.md#drifft.FFT) or [NUFFT](../api/backends.md#drifft.NUFFT)
backend for the destination population size, then bind them in a
[`Transition`](../api/transitions.md).

A transition supports two paired operations: `expect` maps destination values
to their conditional expectations at source states, and `advance` propagates
source probability mass to destination states. It also provides operations
restricted to segregating counts and a solver for expected visits.

Start with the [quickstart](quickstart.md), then read
[transitions and frequency maps](transitions.md) for custom models and
population-size changes. The [model guide](model.md) gives the probability
model and the relationship between expectations and propagation.

## Evolution over a population history

A [`Demography`](../api/demography.md) records population sizes and interval
durations. `DemographicTransition` combines that history with a frequency map,
a backend, and an evolution method.

The backend determines the numerical representation. The method determines
how an interval spanning several generations is computed:

| Component | Choices | Role |
| --- | --- | --- |
| Frequency map | Selection, mutation, composition, or a custom map | Offspring allele probability |
| Backend | FFT, NUFFT, or Jacobi | Representation of the transition |
| Evolution method | Sequential, Polynomial Krylov, or Rational Krylov | Repeated application within an interval |
| Population history | Demography | Sizes and durations of the intervals |

FFT and NUFFT support custom frequency maps. Jacobi currently supports neutral
evolution and symmetric underdominance, and is used with Krylov methods for
forward propagation. Consult [backend tradeoffs](backends.md) for supported
combinations and [constant-size epochs](epochs.md) for method behavior.

See [demographic histories](demography.md) for complete examples. Read
[numerical accuracy](numerics.md) when choosing tolerances and checking
convergence. The Python API reference provides focused pages for each concept's
constructors, operations, and result types.
