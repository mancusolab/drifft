# NUFFT accuracy and derivatives

`expect` samples the Bernstein polynomial at fixed Chebyshev–Lobatto nodes,
computes cosine coefficients using a real-FFT DCT-I, then evaluates them at the
selection probabilities using FINUFFT. The degree and sample geometry depend on
the destination population and accuracy settings, never on the input vector or
selection strength. The backend stores only one node vector.

## Polynomial representation

For destination values $v_j$, let
$B_M[v](q)=\sum_{j=0}^M\binom{M}{j}q^j(1-q)^{M-j}v_j$.
A prepared degree $L$ defines the nodes and samples

$$
\theta_\ell=\frac{\pi\ell}{L},\qquad
q_\ell=\frac{1-\cos\theta_\ell}{2},\qquad
s_\ell\approx B_M[v](q_\ell),\quad \ell=0,\ldots,L.
$$

Sampling uses normalized, truncated binomial stencils. A DCT-I converts these
samples to coefficients. Writing $w_0=w_L=1/2$ and $w_k=1$ otherwise,

$$
c_k=\frac{2w_k}{L}\sum_{\ell=0}^L
w_\ell s_\ell\cos\!\left(\frac{\pi k\ell}{L}\right),
\qquad k=0,\ldots,L.
$$

The interpolating polynomial is

$$
p_L(q)=\sum_{k=0}^L c_kT_k(1-2q)
      =\sum_{k=0}^L c_k\cos(k\theta),\qquad
\theta=2\arcsin\sqrt q=\arccos(1-2q).
$$

A type-2 NUFFT evaluates this series at the requested, generally nonuniform
angles. Its complex Fourier modes are $d_0=c_0$ and
$d_{-k}=d_k=c_k/2$ for $k>0$, so that
$p_L(q)=\operatorname{Re}\sum_{k=-L}^{L}d_ke^{ik\theta}$.
The implementation replaces the values at $q=0$ and $q=1$ by $v_0$ and $v_M$.

!!! note "Fixed geometry, changing coefficients"

    The nodes and degree depend on population size and accuracy controls.
    Samples and coefficients depend linearly on the supplied vector $v$;
    evaluation locations depend on the frequency map.

## Accuracy controls

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
amplified sampling bound.

!!! warning "Accuracy budgets are not error certificates"

    NUFFT tolerance and the roundoff allocation require empirical validation.
    Value-error bounds do not certify derivative accuracy; check gradients
    separately for the intended workload.

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


For selection gradients, differentiate the polynomial in $q$:

$$
\frac{d p_L}{dq}
=-2\sum_{k=1}^L k c_k U_{k-1}(1-2q),
$$

where $U_k$ is a Chebyshev polynomial of the second kind. The implementation
converts this derivative back to first-kind coefficients using a recurrence.
This avoids differentiating the singular angle coordinate at the endpoints.
For a frequency map $q_s$, the chain rule gives
$\partial_s p_L(q_s)=p_L'(q_s)\,\partial_s q_s$.
These are derivatives of the approximation; their error must be assessed
separately from the value-error budget.
