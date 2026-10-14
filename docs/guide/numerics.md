# NUFFT accuracy and derivatives

`expect` samples the Bernstein polynomial at fixed Chebyshev–Lobatto nodes,
computes cosine coefficients using a real-FFT DCT-I, then evaluates them at the
selection probabilities using FINUFFT. The degree and sample geometry depend on
the destination population and accuracy settings, never on the input vector or
selection strength. The backend stores only one node vector.

| Option | Default | Meaning |
| --- | --- | --- |
| `accuracy` | `1e-10` | Nominal value-error budget; `[1e-13, .1)` |
| `tail_tol` | allocated automatically | Optional stricter omitted binomial mass per node |
| `nufft_tol` | `min(1e-13, accuracy/4)` | Requested FINUFFT tolerance; at least `1e-15` |
| `batch_size` | `32` | Sampling nodes processed together |
| `derivative_accuracy` | `False` | Increase degree by 50%, capped at exact degree; heuristic, not a derivative certificate |
| `dtype` | `jnp.float64` | Only float64; enable x64 before planning |

The nominal budget is split equally between spectral interpolation, sampling,
NUFFT evaluation and roundoff. The fixed degree uses the value bound
`4*M/L * exp(-L**2/(2*M))`; at `L=M` interpolation is exact in exact arithmetic.
Per-node tail error is divided by twice a conservative Lobatto Lebesgue constant
`1 + (2/pi)*log(L+1)`. Backend fields expose both the spectral bound and the
amplified sampling bound. NUFFT tolerance and the roundoff allocation are
**not rigorous error certificates**; both require empirical validation.

Sampling builds nonnegative weights from binomial mode-relative ratios, then
normalizes each retained stencil. A Bernstein tail bound controls discarded
mass. This changes a sampling-node expectation by at most twice that mass times
`max(abs(values))`. It is fixed linear preprocessing, not normalization of a
transition output or segregating row. Endpoint samples and transition rows are
exact. Spectral interpolation can still introduce tiny negative coefficients;
outputs are neither clipped nor renormalized.

`advance` is the algebraic transpose of the same approximation. Its sampling
stage regenerates each stencil batch and scatters into one output accumulator.
There is no permanent node-by-stencil weight table. Expected work is
`O(L*W + L*log(L) + source_gene_copies)` up to NUFFT tolerance factors, with
`L` and `W` approximately proportional to `sqrt(M*log(M))`. Working array storage
is `O(M + source_gene_copies + L + batch_size*W)`, excluding library-managed
FINUFFT scratch. XLA memory reports do not measure that external scratch.

Selection derivatives evaluate the derivative polynomial in coefficient space,
including at probability zero and one. The transpose has a matched derivative
rule; it does not differentiate the singular angular coordinate. First and
second polynomial derivatives, propagation gradients and small expected-visits
gradients have independent reference tests. Sampling nodes are fixed geometry:
differentiating the backend's node leaves is rejected. Differentiate selection
parameters or input vectors while holding the backend fixed.
