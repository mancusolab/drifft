# Chebyshev and NUFFT transforms

`NUFFT` samples binomial expectations, converts them to polynomial coefficients,
and evaluates the polynomial at source probabilities. It accelerates a single
transition; the state between generations remains population-sized.

## Problem and representation

Let $M$ be the destination gene-copy count, $N$ the number of source rows, and
$n$ the number of binomial sampling nodes. The input contains $M+1$ values;
the polynomial is evaluated at $N$ offspring probabilities. The retained degree
is set by population size and accuracy controls, rather than by the input vector.

## Computation

### Ordinary Lobatto interpolation

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

### Filtered encoding

Use `NUFFT(M, encoding="filtered")` to taper the sampled cosine coefficients.
With `L` retained modes, it samples `L+1` Lobatto nodes, drops the terminal
mode, and applies weights `min(1, (L-k)/w)` to modes `k=0,...,L-1`.
The taper width is `w = ceil(1/atanh(L/M))`. This remains a fixed linear map
of the input values. `degree` is the output polynomial degree `L-1`, while
`nodes.size` is `L+1`; `taper_width` is the number of tapered modes including
the dropped terminal mode.

Filtering loses endpoint interpolation. Rather than only substituting endpoint
values, the encoding adds the global linear correction

$$
p(q)\leftarrow p(q)+(1-q)[v_0-p(0)]+q[v_M-p(1)].
$$

This changes only the constant and linear Chebyshev coefficients, so the
corrected polynomial retains its degree and its derivatives remain consistent
at the endpoints. Exact endpoint substitutions also suppress evaluation roundoff.

## Cost and reuse

With at most $W$ retained binomial entries per node, forming samples costs
$O(nW)$. The DCT costs $O(n\log n)$. NUFFT evaluation adds a transform cost
that depends on degree, output count $N$, and requested tolerance; producing a
full population-sized output still requires work proportional to its length.
Thus a transform on the retained modes is only one stage of the transition.

Prepared nodes and stencil geometry are reused. Binomial weights are regenerated
in batches, controlled by `batch_size`, rather than retained as a full table.
Sampling workspace depends on batch size and window width; native FINUFFT
workspace is additional to JAX's reported memory. No dense reduced matrix is
powered between generations.

## Accuracy and controls

| Control | Role |
| --- | --- |
| `accuracy` | Nominal budget used to choose polynomial degree |
| `encoding` | Ordinary interpolation or filtered approximation |
| `tail_tol` | Omitted binomial mass at sampling nodes |
| `nufft_tol` | Requested FINUFFT evaluation tolerance |
| `derivative_accuracy` | Larger-degree heuristic for derivatives |
| `batch_size` | Sampling workspace and batching tradeoff |

The default `encoding="lobatto"` uses the interpolation described above.
Its planner takes the smaller of the original Gaussian-tail bound and the
sharper Lobatto bound
`4.2 * (1 + Lambda_d) * exp(-M * Psi((d+1)/M))`, where
`Lambda_d = 1 + 2/pi * log(d+1)`. The latter applies when
`sqrt(M) <= d+1 < M`. The encoding remains ordinary interpolation.

The planner includes the correction in the ideal row-l1 certificate:

$$
\delta\le12\exp[-M\Psi(L/M)],\qquad
\Psi(u)=\tfrac12[(1+u)\log(1+u)+(1-u)\log(1-u)].
$$

It searches `ceil(sqrt(M)) <= L < M`; if compression cannot meet the spectral
budget, it uses the untapered degree-M representation (`taper_width=0`).
Sampling error includes both stencil normalization and endpoint correction:
`sampling_bound = 4 * tail_tol * min((2L-w)/w, 5+log(L/w))` in the compressed
case. These bounds exclude floating-point and actual NUFFT evaluation errors.
A FINUFFT tolerance is not a certified operator-norm error: coefficient norms
can amplify it. Neither encoding guarantees positivity or derivative accuracy.

The derivation is in the [filtered Chebyshev note](https://github.com/mancusolab/drifft/blob/main/notes/wf_filtered_chebyshev_bounds.md). Its lower bound concerns
uniform approximation of the full binomial family by bounded-degree
polynomials, not arbitrary low-rank factorizations or Jacobi's implementation.
Filtered encoding still uses population-sized epoch states, like ordinary
NUFFT; it does not add Jacobi-style reduced-state powering.

## Differentiation and adjoints


For selection gradients, differentiate the polynomial in $q$:

$$
\frac{d p_L}{dq}
=-2\sum_{k=1}^L k c_k U_{k-1}(1-2q),
$$

where $U_k$ is a Chebyshev polynomial of the second kind. For a frequency map
$q_s$, the chain rule gives
$\partial_s p_L(q_s)=p_L'(q_s)\,\partial_s q_s$.
This avoids differentiating the singular angle coordinate at the endpoints.
These are derivatives of the approximation, so their accuracy still needs
separate validation.

The approximation is linear in the supplied values. Forward mass propagation
uses its matched transpose. Changing the frequency map changes evaluation
locations; changing the input values changes sampled coefficients. Derivatives
of prepared sampling nodes are unsupported. Backend geometry is prepared outside
JIT and differentiation.

## Use in drifft

Bind `NUFFT` to a frequency map with `Transition`. It supports general selection
and mutation, rectangular transitions, expected visits, and demographic evolution
with Sequential or either Krylov method. The [algorithm guide](../guide/backends.md)
compares the choices; the [accuracy guide](../guide/numerics.md) gives validation
examples, and the [NUFFT API](../api/backends.md#drifft.NUFFT) lists defaults.
