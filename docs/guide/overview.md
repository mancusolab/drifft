# Overview

`drifft` computes how allele-count distributions change under the discrete-time
Wright–Fisher model, including selection, mutation, and changing population
sizes. It also computes expectations and expected visits before absorption,
and supports differentiation through biological parameters.

The name brings together genetic drift and the FFT. The computational problem
is a binomial transition: each possible parental allele count gives a row of
probabilities for offspring counts. A dense transition has quadratically many
entries in the population gene-copy count. Nearby offspring probabilities
produce similar rows, giving structure that can be exploited without storing
the full matrix.

## Where the speedups come from

FFT and NUFFT accelerate individual transitions. FFT factors local portions of
the binomial kernel into scaled convolutions; NUFFT samples binomial
expectations, represents them by a polynomial, and evaluates that polynomial
with a fast transform. These are different numerical representations of the
same biological transition.

Jacobi represents evolution in a smaller polynomial state. Krylov methods can
then approximate evolution over a long constant-size interval without applying
a transition once per generation. Preparation and factorization have costs of
their own; the saving depends on interval length, accuracy, and reuse.

!!! note "Compression and transforms have different costs"

    A dense reduced matrix with $r$ coefficients still costs $O(r^2)$ to apply.
    An FFT on $r$ values costs $O(r\log r)$, but does not apply an arbitrary
    reduced matrix. Encoding and decoding a population-sized vector add work.

The [algorithm guide](backends.md) explains when each approach is useful.
[Benchmark reports](../benchmarks.md) give measurements with their workloads
and accuracy checks; no backend is fastest for every problem.

## How a calculation fits together

| Component | What you specify | Role |
| --- | --- | --- |
| Frequency map | Selection, mutation, composition, or a custom map | Offspring allele probability |
| Backend | FFT, NUFFT, or Jacobi | Numerical representation of the transition |
| Evolution method | Sequential, Polynomial Krylov, or Rational Krylov | Repeated evolution within a constant-size interval |
| Population history | Sizes and interval durations | Sequence of demographic changes |

For one generation, bind a frequency map and FFT or NUFFT backend in
`Transition`. Its `advance` method propagates a distribution; `expect` computes
conditional expectations using the transpose relationship. For a history,
`DemographicTransition` combines `Demography` with a map, backend, and method.

## Reading the guide

[Install the package](installation.md), then run the [quickstart](quickstart.md).
The [model chapter](model.md) introduces the probability assumptions. Continue
with [selection and mutation](frequency_maps.md), [transition calculations](transitions.md),
and [population histories](demography.md) for practical examples.

Choose [numerical algorithms](backends.md) for your workload, then use
[accuracy and differentiation](numerics.md) to validate the result and its
gradients. [Expected visits](solves.md) covers occupation quantities through
absorption. Detailed derivations live in Numerical methods; signatures and
return types live in the Python API reference.
