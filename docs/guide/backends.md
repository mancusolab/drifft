# Choosing numerical algorithms

Start with FFT for one-generation calculations and FFT with Sequential for
population histories. This supports all frequency maps, forward propagation,
and conditional expectations. Keep that configuration until its time or memory
cost becomes a limitation.

A backend represents one generation; an evolution method applies that
representation across an interval. The [computational overview](../methods/overview.md)
explains the distinction and the errors each introduces.

## Alternatives

Switch algorithms to address a measured cost in your calculation. These choices
offer different tradeoffs; check compatibility before comparing performance.

| Calculation | Alternative |
| --- | --- |
| Less retained transition geometry | NUFFT, which trades storage for computation |
| Faster propagation across long constant-size intervals | Jacobi with RationalKrylov, which works in a reduced state |
| Krylov propagation with a custom frequency map | PolynomialKrylov with FFT or NUFFT |
| Independent checks on small populations | Dense, which explicitly forms binomial rows and uses ordinary JAX differentiation |

Jacobi's reduced Krylov assembly supports a bounded family of diploid selection
and mutation maps; check [compatibility and parameter bounds](../api/backends.md#drifft.Jacobi)
before switching. Its one-generation operations and Sequential support arbitrary
frequency maps producing valid probabilities. If you need `model.expect()`,
use Sequential with any backend.

The [Schiffels–Durbin example](../examples/schiffels_durbin.md) demonstrates
Jacobi and RationalKrylov with selection and recurrent mutation. The
[numerical methods](../methods/jacobi.md) explain how the reduction works.

## Validation

Compare the final quantity you care about against your original configuration,
then tighten the numerical settings and compare again. Agreement of a mean
alone can hide errors in probabilities or gradients. See
[accuracy and differentiation](numerics.md) for the validation procedure.

Use the [timing guidance](numerics.md#memory-and-timing) to compare first-call
and repeated-use costs consistently.
