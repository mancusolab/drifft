# Jacobi reduction

`Jacobi` approximates binomial expectations in polynomial coefficients. It
supports one-generation transitions and supplies a reduced state for Krylov
evolution. This differs from the Chebyshev interpolation used by `NUFFT`.

Configuration: [API reference](../api/backends.md#drifft.Jacobi).

## Problem and representation

Jacobi separates endpoint values from an interior polynomial expansion.
This supplies both a one-generation expectation approximation and a reduced
state for evolution across a constant-size interval.

Let $M$ be the gene-copy count and $r$ the retained interior polynomial count.
The full count-space vector has $M+1$ entries; the reduced representation
has $r+2$ coordinates, or $r$ when retaining only segregating states.

## Computation

Repeated count-space evolution processes a population-sized vector at every
step. A dense transition costs $O(M^2)$ per application; FFT and NUFFT reduce
that cost but still operate on population-sized states. Jacobi instead
factors the transition through polynomial coordinates, allowing Krylov to
work with a reduced matrix whose products cost $O(r^2)$. When $r\ll M$,
long intervals can amortize the cost of encoding and decoding the state.
This saving applies to reduced Krylov evolution; Sequential repeats the
count-space transforms each generation.

For destination values $f_j$, separate the endpoint interpolant and expand
the remaining binomial expectation as

$$
B_M[f](q)\approx(1-q)f_0+qf_M
+(1-\xi^2)\sum_{k=0}^{r-1}c_k\phi_k(\xi),\qquad \xi=2q-1.
$$

Here $\phi_k$ are Jacobi polynomials with parameters $(1,1)$, orthonormal for
weight $3(1-\xi^2)/4$ on $[-1,1]$. The envelope makes the interior basis vanish
at both endpoints, preserving $f_0$ and $f_M$ explicitly.

Write the expectation approximation as $\widetilde P=DE$, where $E$ maps
values to coefficients and $D$ evaluates at offspring probabilities after
the full frequency map. Adjacent encoding and evaluation maps combine into
the reduced matrix $H_{\mathrm J}=ED$, so in exact coefficient arithmetic,

$$
\widetilde P^{\,g}=D(ED)^{g-1}E=DH_{\mathrm J}^{g-1}E,\qquad g\ge1.
$$

Only the outer maps act on count-space vectors; the intervening power stays
in reduced coordinates. Forward mass propagation uses the transpose of this
factorization: encode once, evolve the reduced state, and decode once.
See [Krylov methods](jacobi_krylov.md) for the power approximation and
source mass recurrence.

Gauss–Jacobi quadrature with $n_{\mathrm{quad}}=r+32$ nodes and truncated,
normalized binomial stencils constructs the coefficient transform. For
supported frequency maps, rational functions of a symmetric tridiagonal Hahn
matrix construct the reduced operator. A padded
$r_{\mathrm{pad}}\times r_{\mathrm{pad}}$ block, banded solves, and polynomial
recurrences produce the retained matrix and endpoint coupling.

!!! note "Projection and assembly theory"

    The [analysis note](https://github.com/mancusolab/drifft/blob/main/notes/wf_jacobi_bounds.md)
    distinguishes ideal projection from sampled coefficients and padded
    matrix assembly, and develops their truncation bounds.

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
