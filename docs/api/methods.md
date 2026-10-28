# Evolution methods

`AbstractEpochMethod` defines how a prepared operator is applied over a
constant-size interval. Implementations are `Sequential`,
`PolynomialKrylov`, and `RationalKrylov`.
`EpochResult` and `EvolutionStatus` describe the returned distribution and
convergence status. See [constant-size epochs](../methods/epochs.md) for interval
timing and diagnostics.

::: drifft.AbstractEpochMethod
    options:
        members: [validate, solve, advance]

::: drifft.EpochResult
    options:
        members: []

::: drifft.EvolutionStatus
    options:
        members: []

## Sequential

Apply the prepared transition generation by generation. See
[constant-size epochs](../methods/epochs.md) for timing and immigration semantics.

::: drifft.Sequential
    options:
        members: [__init__, advance]

## Polynomial Krylov

Approximate repeated evolution using polynomial Krylov projection. See
[constant-size epochs](../methods/epochs.md) for convergence and differentiation
limits. Results use the shared [method result types](#drifft.EpochResult).

::: drifft.PolynomialKrylov
    options:
        members: [__init__, advance]

## Rational Krylov

Approximate repeated evolution using rational Krylov projection and shifted
linear solves. See [constant-size epochs](../methods/epochs.md) for convergence
and differentiation limits. Results use the shared [method result types](#drifft.EpochResult).

::: drifft.RationalKrylov
    options:
        members: [__init__, advance]
