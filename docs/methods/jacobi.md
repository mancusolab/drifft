# Jacobi reduction

`Jacobi` approximates binomial expectations in polynomial coefficients. It
supports one-generation transitions and supplies a reduced state for Krylov
evolution. This differs from the Chebyshev interpolation used by `NUFFT`.

For configuration, see the [API reference](../api/backends.md#drifft.Jacobi); for a workflow, see
[choosing algorithms](../guide/backends.md). Notation follows the
[shared conventions](../guide/notation.md). The
[computational overview](overview.md) places this method within the full calculation.

## Problem and representation

Jacobi separates the endpoint values from an interior polynomial expansion.
This provides a small state that Krylov methods can evolve across many generations.

Let $M$ be the gene-copy count and $r$ the retained interior polynomial count.
For count-space values $f_j$, separate the linear interpolant between $f_0$
and $f_M$. Approximate the remaining binomial expectation in

$$
(1-\xi^2)\phi_k(\xi),\qquad k=0,\ldots,r-1,\qquad \xi=2q-1,
$$

where $\phi_k$ are normalized Jacobi polynomials with parameters $(1,1)$,
orthonormal for weight $3(1-\xi^2)/4$ on $[-1,1]$. The envelope makes interior
basis functions vanish at the count endpoints. The full representation
has $r+2$ coordinates; the segregating representation has $r$.

Write the one-generation expectation approximation as $\widetilde P=DE$, with
$E$ mapping values to coefficients and $D$ evaluating at the offspring
probabilities after the full frequency map. One-generation operations apply
these external maps directly. For Krylov evolution, $H_{\mathrm J}=ED$ is a reduced square
matrix. In exact coefficient arithmetic,

$$
\widetilde P^{\,g}=DH_{\mathrm J}^{g-1}E,\qquad g\ge1.
$$

The [analysis note](https://github.com/mancusolab/drifft/blob/main/notes/wf_jacobi_bounds.md)
defines the ideal projection and distinguishes it from sampled coefficients
and padded matrix assembly.

## Computation

The backend constructs external transforms for entering and leaving coefficient
space, plus a reduced operator for evolution within it. Reduced assembly avoids
forming a population-sized matrix.

!!! note "Advanced: reduced operator assembly"

    The construction below uses orthogonal-polynomial quadrature and rational
    functions of a tridiagonal coordinate matrix. The linked analysis note
    develops the padding and truncation bounds.

Gauss–Jacobi quadrature with $n_{\mathrm{quad}}=r+32$ nodes and truncated,
normalized binomial stencils constructs the coefficient transform. For
supported frequency maps, rational functions of a symmetric tridiagonal Hahn
matrix construct the reduced operator. A padded
$r_{\mathrm{pad}}\times r_{\mathrm{pad}}$ block, banded solves, and polynomial
recurrences produce the retained matrix and endpoint coupling.

Krylov encodes source mass once, evolves the reduced state, and decodes once.
Sequential instead repeats the full count-space transition, including the
external transforms at each generation. See [Krylov methods](jacobi_krylov.md)
for the power approximation and immigration recurrence.

## Cost and reuse

Geometry depends on population size, rank, and numerical settings. It is
reusable across supported selection and mutation parameters within the planned
bounds. The reduced matrix depends on both selection and mutation.
RationalKrylov factors its shifted matrix once per power action and reuses that
factorization across basis vectors. The shift also depends on interval duration.

| Stage | Cost or storage |
| --- | --- |
| Reduced assembly | $O(r r_{\mathrm{pad}})$ arithmetic |
| Reduced matrix storage and matvec | $O(r^2)$ |
| Dense LU for rational Krylov | $O(r^3)$ factorization; $O(r^2)$ per solve |
| Source-mass encoding | $O(Mr)$ with the streamed polynomial recurrence |
| Decoding | $O(n_{\mathrm{quad}}r+n_{\mathrm{quad}}W+M)$ for stencil width $W$ |
| One-generation expectation at $n_{\mathrm{eval}}$ probabilities | $O(n_{\mathrm{quad}}r+n_{\mathrm{quad}}W+M+n_{\mathrm{eval}}r)$ |
| Stored quadrature geometry | $O(n_{\mathrm{quad}}r+n_{\mathrm{quad}}W+M)$, with symmetry reducing constants |

With Krylov, encoding and decoding are paid at the interval boundaries rather
than for every reduced step. A reduced matvec is dense, not an FFT. Benefits
depend on how much repeated population-space work the reduction avoids and how well setup is
amortized. Finite-tolerance rank bounds also contain accuracy and population-size
factors; do not interpret $r$ as an unconditional constant times $\sqrt M$.

## Accuracy and controls

Rank controls the polynomial representation, while assembly padding controls
the reduced operator. These errors remain separate from the subsequent Krylov
power approximation.

| Control | Role |
| --- | --- |
| `rank` | Retained interior polynomial modes |
| `one_step_tolerance` | Plans ideal truncation and assembly padding separately |
| `max_strength` | Static absolute bound on both diploid coefficients, used to plan padding |

The `max_strength` bound and supported-map restrictions apply to reduced Krylov
assembly. One-generation operations and Sequential evaluate the frequency map
directly and accept any map producing valid probabilities.

The default rank certificate covers ideal one-step projection. Quadrature,
binomial stencil truncation, padded assembly, and arithmetic add further errors.
Increase rank and compare with a small Dense reference or a tightened FFT/NUFFT
calculation, following the [accuracy guide](../guide/numerics.md).

## Differentiation and adjoints

The external transforms have matched transposes, and their derivative rules
stream polynomial recurrences to limit memory. Model parameters and mass are
differentiable; rank, padding, quadrature nodes, and population size are fixed.

The full-state construction preserves mass but need not preserve positivity.
Endpoint rows are absorbing only when the frequency map fixes zero and one;
segregating evolution discards endpoint mass on every generation regardless.

Sequential preserves the expectation/propagation pairing. An input-dependent
Krylov approximation does not provide a matched expectation transpose; use
Sequential for `model.expect()`. The [autodiff chapter](autodiff.md) explains
parameter derivatives through the transforms and the reduced power.
