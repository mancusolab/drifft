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

## Epoch backends

See [constant-size epochs](../guide/epochs.md) for timing, immigration,
approximation diagnostics, and differentiation limits.

??? abstract "`drifft.AbstractEpochBackend`"

    ::: drifft.AbstractEpochBackend
        options:
            members: [advance]

::: drifft.Sequential
    options:
        members: [__init__]

::: drifft.RationalKrylov
    options:
        members: [__init__]

::: drifft.EpochResult
    options:
        members: []
