# Backends

Construct backends outside JIT and reuse them across selection strengths.
See [backend tradeoffs](../guide/backends.md) and
[NUFFT accuracy settings](../guide/numerics.md).

??? abstract "`drifft.AbstractTransitionBackend`"

    ::: drifft.AbstractTransitionBackend
        options:
            members: [expect]

::: drifft.FFT
    options:
        members: [__init__]

::: drifft.NUFFT
    options:
        members: [__init__]

::: drifft.Jacobi
    options:
        members: [__init__]

## Epoch methods

See [constant-size epochs](../guide/epochs.md) for timing, immigration,
approximation diagnostics, and differentiation limits.

??? abstract "`drifft.AbstractEpochMethod`"

    ::: drifft.AbstractEpochMethod
        options:
            members: [validate, solve, advance]

::: drifft.Sequential
    options:
        members: [__init__, advance]

::: drifft.RationalKrylov
    options:
        members: [__init__, advance]

::: drifft.EpochResult
    options:
        members: []

::: drifft.EvolutionStatus
    options:
        members: []
