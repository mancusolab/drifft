# Choosing a backend

A backend prepares a numerical representation for a population size. A method
uses that representation to evolve a distribution over several generations.
Prepare backends outside JIT and reuse them across selection strengths.

| Backend | How it works | Use with |
| --- | --- | --- |
| `FFT` | Local FFT convolutions with interpolation | `Transition`, or `DemographicTransition` with `Sequential` |
| `NUFFT` | A global polynomial evaluated by a nonuniform FFT | `Transition`, or `DemographicTransition` with `Sequential` |
| `Jacobi` | A reduced polynomial representation for repeated evolution | `DemographicTransition` with `RationalKrylov` |

## FFT or NUFFT

Start with FFT for repeated one-generation operations. It caches more geometry
to reduce work during application. NUFFT retains less geometry and regenerates
sampling weights, trading memory for extra work. Measure both on your intended
population sizes and hardware; neither is universally faster.

Both approximate the same binomial transition and support custom frequency
maps, population-size changes, expectations, and expected visits. `Transition`
binds the backend to a frequency map. Its `advance` operation is the transpose
of its `expect` operation, so the two use the same numerical approximation.

FFT's `tail_tol` controls omitted binomial tails; `interp_order` controls local
interpolation. NUFFT's `accuracy` sets a nominal value-error budget; see
[NUFFT accuracy and derivatives](numerics.md) for its controls. Neither backend
clips negative coefficients or renormalizes outputs. Check accuracy and gradients
for your workload, especially near probability endpoints or over long histories.

## Jacobi for long constant-size intervals

Jacobi compresses evolution into a smaller polynomial representation.
`RationalKrylov` then approximates many generations within that representation.
This combination is experimental and currently supports only `Neutral` and
`Underdominance`. It supports forward evolution, but not `model.expect()`.

Use `backend=Jacobi`, `method=RationalKrylov()`, and an explicit FFT or NUFFT
`boundary_backend` for demographic size changes. Jacobi's `rank` controls the
representation; the method's `dimension` controls the power approximation.
Increasing one does not remove errors from the other.

See [constant-size methods](epochs.md) for examples and convergence checks.
RationalKrylov currently requires Jacobi; it cannot be paired with FFT or NUFFT.

## Mathematical and API details

The [model guide](model.md) defines the shared binomial transition. The
[backend API reference](../api/backends.md) explains each representation's
mathematics and constructor settings. FFT and NUFFT implement
`AbstractTransitionBackend`; computation methods implement `AbstractEpochMethod`.
