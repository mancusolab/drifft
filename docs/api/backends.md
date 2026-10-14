# Backends

Construct backends outside JIT and reuse them across selection strengths.
See [backend tradeoffs](../guide/backends.md) and
[NUFFT accuracy settings](../guide/numerics.md).

::: drifft.AbstractTransitionBackend
    options:
        members: [expect]

::: drifft.FFT
    options:
        members: [__init__, expect]

::: drifft.NUFFT
    options:
        members: [__init__, expect]
