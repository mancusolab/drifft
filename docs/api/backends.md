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
`DemographicTransition` and a Krylov method. Supports neutral evolution, diploid
selection and its constrained classes, mutation, and one mutation map composed
with one selection rule in either order, within its selection bounds.
Jacobi requires `PolynomialKrylov` or `RationalKrylov`; it is not accepted by
`Transition` or `Sequential`. For demographic evolution, supply an explicit
FFT or NUFFT boundary backend.
See [backend tradeoffs](../guide/backends.md)
and [constant-size epochs](../methods/epochs.md).

::: drifft.Jacobi
    options:
        members: [__init__]
