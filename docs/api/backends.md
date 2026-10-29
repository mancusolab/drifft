# Numerical backends

`AbstractTransitionBackend` defines the expectation operation used by
`Transition`. Concrete implementations are `FFT` and `NUFFT`.
`Jacobi` supplies a separate reduced representation for demographic
evolution. See [backend tradeoffs](../guide/backends.md) for choosing a backend.

::: drifft.AbstractTransitionBackend
    options:
        members: [expect]

## FFT

Local FFT convolutions with interpolation. Construct geometry outside JIT
and reuse it across selection parameters. See [backend tradeoffs](../guide/backends.md).

::: drifft.FFT
    options:
        members: [__init__]

## NUFFT

Global polynomial evaluation using a nonuniform FFT. See
[backend tradeoffs](../guide/backends.md) and [accuracy controls](../guide/numerics.md).

::: drifft.NUFFT
    options:
        members: [__init__]

## Jacobi

Reduced Jacobi polynomial geometry for repeated evolution. Use with
`DemographicTransition` or direct epoch operations.

| Frequency map | Supported domain |
| --- | --- |
| `Neutral` | No parameters |
| `DiploidSelection` and its constrained classes | Both derived coefficients finite, with absolute value at most `max_strength` |
| `Mutation` | Each rate finite and in `[0, 1]` |
| `Composition` | Exactly one mutation map and one supported selection rule, in either order |

`max_strength` defaults to `0.1` and must lie in `[0, 1)`. Named
`Underdominance` and `Overdominance` rules also require nonnegative strength,
including in compositions. Process domains are checked at runtime even for
zero-generation calls. Custom maps and longer compositions are not supported.

Jacobi requires `PolynomialKrylov` or `RationalKrylov`; it is not accepted by
`Transition` or `Sequential`. For demographic evolution, supply an explicit
FFT or NUFFT boundary backend.
See [backend tradeoffs](../guide/backends.md) and
[constant-size epochs](../methods/epochs.md).

::: drifft.Jacobi
    options:
        members: [__init__]
