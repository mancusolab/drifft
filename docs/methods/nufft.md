# Chebyshev and NUFFT transforms

`NUFFT` samples binomial expectations, converts them to polynomial coefficients,
and evaluates the polynomial at source probabilities. It accelerates a single
transition; the state between generations remains population-sized.

For configuration, see the [API reference](../api/backends.md#drifft.NUFFT); for a workflow, see
[choosing algorithms](../guide/backends.md). Notation follows the
[shared conventions](../guide/notation.md). The
[computational overview](overview.md) places this method within the full calculation.

## Problem and representation

Binomial expectations vary smoothly with offspring frequency. NUFFT represents
that dependence by one polynomial and evaluates it at all requested probabilities.

Let $M$ be the destination gene-copy count, $n_{\mathrm{eval}}$ the number of source rows, and
$n_{\mathrm{nodes}}$ the number of binomial sampling nodes. The input contains $M+1$ values;
the polynomial is evaluated at $n_{\mathrm{eval}}$ offspring probabilities. The retained degree
is set by population size and accuracy controls, rather than by the input vector.

## Computation

Both encodings sample binomial expectations and transform the samples into
Chebyshev coefficients. They differ in whether the polynomial interpolates
the samples or filters its highest modes.

### Ordinary Lobatto interpolation

Lobatto interpolation chooses nodes that permit a fast cosine transform.
The resulting coefficients describe the expectation between those nodes.

For destination values $f_j$, let
$B_M[f](q)=\sum_{j=0}^M\binom{M}{j}q^j(1-q)^{M-j}f_j$.
A prepared degree $L$ defines the nodes and samples

$$
\vartheta_\ell=\frac{\pi\ell}{L},\qquad
q_\ell=\frac{1-\cos\vartheta_\ell}{2},\qquad
s_\ell\approx B_M[f](q_\ell),\quad \ell=0,\ldots,L.
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
      =\sum_{k=0}^L c_k\cos(k\vartheta),\qquad
\vartheta=2\arcsin\sqrt q=\arccos(1-2q).
$$

A type-2 NUFFT evaluates this series at the requested, generally nonuniform
angles. Its complex Fourier modes are $d_0=c_0$ and
$d_{-k}=d_k=c_k/2$ for $k>0$, so that
$p_L(q)=\operatorname{Re}\sum_{k=-L}^{L}d_ke^{ik\vartheta}$.
The implementation replaces the values at $q=0$ and $q=1$ by $f_0$ and $f_M$.

!!! note "Fixed geometry, changing coefficients"

    The nodes and degree depend on population size and accuracy controls.
    Samples and coefficients depend linearly on the supplied vector $f$;
    evaluation locations depend on the frequency map.

### Filtered encoding

Filtered encoding tapers the highest cosine modes to control uniform
approximation error. Select it with `NUFFT(M, encoding="filtered")`; unlike
ordinary interpolation, it needs an endpoint correction.
Write $n_{\mathrm{modes}}=L+1$ for the retained coefficient count, so $L$
continues to denote the output degree. Filtered encoding samples
$n_{\mathrm{nodes}}=n_{\mathrm{modes}}+1=L+2$ Lobatto nodes, drops the terminal
mode, and applies weights
$\min(1,(n_{\mathrm{modes}}-j)/w)$ to modes $j=0,\ldots,L$.
The taper width is $w=\lceil1/\operatorname{atanh}(n_{\mathrm{modes}}/M)\rceil$.
This remains a fixed linear map of the input values. The API's `degree` is $L$,
`nodes.size` is $L+2$, and `taper_width` counts tapered modes including the
dropped terminal mode.

Filtering loses endpoint interpolation. Rather than only substituting endpoint
values, the encoding adds the global linear correction

$$
p(q)\leftarrow p(q)+(1-q)[f_0-p(0)]+q[f_M-p(1)].
$$

This changes only the constant and linear Chebyshev coefficients, so the
corrected polynomial retains its degree and its derivatives remain consistent
at the endpoints. Exact endpoint substitutions also suppress evaluation roundoff.

## Cost and reuse

The main costs are sampling the binomial rows and evaluating the polynomial.
The sampled geometry is reusable, but coefficients change with the input values.

With at most $W$ retained binomial entries per node, forming samples costs
$O(n_{\mathrm{nodes}}W)$. The DCT costs $O(n_{\mathrm{nodes}}\log n_{\mathrm{nodes}})$. NUFFT evaluation adds a transform cost
that depends on degree, output count $n_{\mathrm{eval}}$, and requested tolerance; producing a
full population-sized output still requires work proportional to its length.
Thus a transform on the retained modes is only one stage of the transition.

Prepared nodes and stencil geometry are reused. Binomial weights are regenerated
in batches, controlled by `batch_size`, rather than retained as a full table.
Sampling workspace depends on batch size and window width; native FINUFFT
workspace is additional to JAX's reported memory. No dense reduced matrix is
powered between generations.

## Accuracy and controls

Polynomial truncation, binomial sampling, and NUFFT evaluation introduce
different errors. The controls below adjust those stages separately.

| Control | Role |
| --- | --- |
| `accuracy` | Nominal budget used to choose polynomial degree |
| `encoding` | Ordinary interpolation or filtered approximation |
| `tail_tol` | Omitted binomial mass at sampling nodes |
| `nufft_tol` | Requested FINUFFT evaluation tolerance |
| `derivative_accuracy` | Larger-degree heuristic for derivatives |
| `batch_size` | Sampling workspace and batching tradeoff |

Neither encoding guarantees positivity or derivative accuracy. A FINUFFT
tolerance is not an operator-error certificate: coefficient norms can amplify
its effect. Validate final quantities with the [accuracy guide](../guide/numerics.md).

!!! note "Advanced: spectral bounds"

    The planner sizes the polynomial using ideal uniform-error bounds. For
    filtered encoding, the corrected row error satisfies
    $\delta\le12\exp[-M\Psi(n_{\mathrm{modes}}/M)]$, where
    $\Psi(u)=\tfrac12[(1+u)\log(1+u)+(1-u)\log(1-u)]$.
    If compression cannot meet the budget, it uses the untapered degree-$M$
    representation. These certificates exclude floating-point and actual NUFFT
    errors; the [analysis note](https://github.com/mancusolab/drifft/blob/main/notes/wf_filtered_chebyshev_bounds.md)
    gives the interpolation, filtering, and sampling bounds and their assumptions.

## Differentiation and adjoints

Differentiating in offspring probability avoids the singular angular coordinate
at the endpoints. The resulting gradient is the derivative of the represented
polynomial, so its accuracy must still be checked separately.

For selection gradients, differentiate the polynomial in $q$:

$$
\frac{d p_L}{dq}
=-2\sum_{k=1}^L k c_k U_{k-1}(1-2q),
$$

where $U_k$ is a Chebyshev polynomial of the second kind. For a frequency map
$q_s$, the chain rule gives
$\partial_s p_L(q_s)=p_L'(q_s)\,\partial_s q_s$.
The approximation is linear in the supplied values. Forward mass propagation
uses its matched transpose. Changing the frequency map changes evaluation
locations; changing the input values changes sampled coefficients. Derivatives
of prepared sampling nodes are unsupported. Backend geometry is prepared outside
JIT and differentiation.
