# FFT interpolation on JAX 0.12

This CPU experiment measures the cost of changing the interpolation weights in
the full FFT expectation, angular JVP (value and tangent together), and transpose
actions on JAX and jaxlib 0.12.0.1. The baseline uses the production Lagrange products. The candidate forms
the same cardinal polynomials in exact rational arithmetic at trace time, rounds
the coefficients to float64, and evaluates all weights with
`jax.lax.polynomial` and `unroll=16`. Its argument is the fractional cell
position in [0,1], with stencil offsets −5 through 6 for order 12.

| Gene copies | Operation | Product (ms) | Polynomial (ms) |
| ---: | --- | ---: | ---: |
| 8,192 | Expectation | 1.543 | 1.646 |
| 8,192 | Angular JVP | 1.769 | 1.672 |
| 8,192 | Transpose | 2.623 | 2.499 |
| 32,768 | Expectation | 5.225 | 5.017 |
| 32,768 | Angular JVP | 6.452 | 5.543 |
| 32,768 | Transpose | 10.236 | 9.686 |

Each entry is the pooled median of 50 synchronized warm calls. Two serial
processes per implementation ran in opposite order across rounds. Operations
were shuffled within each repetition. Each call used a full evenly spaced
probability grid including endpoints and a seeded signed normal vector.
The input vector was drawn with NumPy’s default generator and seed 42.
The transpose used the same signed vector as its cotangent.
Compilation and preparation are excluded. The angular tangent was
`sqrt(q * (1 - q))`. Plans used `atol=1e-12`, `grad_atol=1e-10` and angular
sensitivity one. [Timing samples and environment](fft_lax_polynomial_cpu.json).

The polynomial variant passed the independent Decimal row and angular-gradient
tests. At 8,192 and 32,768 copies, the maximum sampled row L1 errors were
8.95e-14 and 2.65e-14. The corresponding angular-gradient errors were 8.69e-12
and 2.66e-11. These results cover order 12 on one CPU.

## Larger grids and longer timing runs

The same workload ran at 65,536 and 131,072 gene copies, with 100 synchronized
warm calls per operation in each of three fresh processes per implementation.
The paired rounds ran product first, then polynomial first, then product first.
Each table entry is a pooled median of 300 calls. The last column gives the
range of changes in the three paired round medians (negative means less time).

| Gene copies | Operation | Product (ms) | Polynomial (ms) | Paired change |
|---:|---|---:|---:|---:|
| 65,536 | Expectation | 14.12 | 13.98 | −3.3% to +5.2% |
| 65,536 | Angular JVP | 16.96 | 14.73 | −15.0% to −8.1% |
| 65,536 | Transpose | 25.92 | 24.26 | −8.7% to +1.3% |
| 131,072 | Expectation | 25.69 | 25.60 | −1.4% to +12.6% |
| 131,072 | Angular JVP | 32.63 | 28.09 | −15.5% to −6.5% |
| 131,072 | Transpose | 48.64 | 46.69 | −5.4% to +6.4% |

Only the JVP improved in every paired round at both sizes. Absolute timings
drifted across processes. On these signed-vector inputs, the largest absolute
difference between implementations was 2.23e-16 for primal or transpose output
and 4.78e-12 for an angular tangent. The independent Decimal comparisons above
cover the smaller sizes. [Large-run samples and per-round medians](fft_lax_polynomial_large_cpu.json).
