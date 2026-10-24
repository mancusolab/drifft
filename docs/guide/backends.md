# Choosing a backend

A backend prepares a numerical representation for a population size. A method
uses that representation to evolve a distribution over several generations.
Prepare backends outside JIT and reuse them across selection strengths.

| Backend | How it works | Use with |
| --- | --- | --- |
| `FFT` | Local FFT convolutions with interpolation | `Transition`, or `DemographicTransition` with Sequential or either Krylov method |
| `NUFFT` | A global polynomial evaluated by a nonuniform FFT | `Transition`, or `DemographicTransition` with Sequential or either Krylov method |
| `Jacobi` | A reduced polynomial representation for repeated evolution | `DemographicTransition` with `PolynomialKrylov` or `RationalKrylov` |

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

NUFFT offers `encoding="lobatto"` (the default) and `encoding="filtered"`.
The filtered variant tapers cosine coefficients and restores endpoints with a
linear polynomial correction. Its tighter spectral certificate can reduce the
number of samples, but does not guarantee a faster transform or more accurate
derivatives. Compare both with `benchmarks/compare_encodings.py`; this measures
fresh-process and warm costs separately and can check rows and repeated steps
against binomial references.

## Jacobi for long constant-size intervals

Jacobi compresses evolution into a smaller polynomial representation.
`RationalKrylov` then approximates many generations within that representation.
This combination currently supports only `Neutral` and
`Underdominance`. It supports forward evolution, but not `model.expect()`.

Use `backend=Jacobi`, `method=RationalKrylov()`, and an explicit FFT or NUFFT
`boundary_backend` for demographic size changes. Jacobi's `rank` controls the
representation; the method's `dimension` controls the power approximation.
Increasing one does not remove errors from the other.

See [constant-size methods](epochs.md) for examples and convergence checks.
RationalKrylov also accepts FFT and NUFFT. Those backends use internal,
unpreconditioned GMRES shifted solves instead of Jacobi's dense LU. Each basis
vector can require many transition applications, so this is not automatically
faster than Sequential. Failed inner solves are reported in the result status.

## Polynomial Krylov across backends

`PolynomialKrylov` works with FFT, NUFFT, or Jacobi using only applications of
the prepared operator. It avoids shifted linear solves, but its basis can need
more vectors than RationalKrylov for long intervals. With FFT/NUFFT, basis
storage grows with population size times Krylov dimension.

Use `method=PolynomialKrylov(dimension=64)` in a demographic model. Check
convergence and compare against Sequential before relying on a speedup.
Frequency-map support follows the backend; Jacobi retains its restrictions.
Like RationalKrylov, this method supports forward evolution only.

## Mathematical and API details

The [model guide](model.md) defines the shared binomial transition. The
[backend API reference](../api/backends.md) explains each representation's
mathematics and constructor settings. FFT and NUFFT implement
`AbstractTransitionBackend`; computation methods implement `AbstractEpochMethod`.
