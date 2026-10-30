# Jacobi reduction

`Jacobi` approximates binomial expectations in polynomial coefficients. It
supports one-generation transitions and supplies a reduced state for Krylov
evolution. This differs from the Chebyshev interpolation used by `NUFFT`.

## Problem and representation

Let $M$ be the gene-copy count and $r$ the retained interior polynomial count.
For count-space values $f_j$, separate the linear interpolant between $f_0$
and $f_M$. Approximate the remaining binomial expectation in

$$
(1-z^2)\phi_k(z),\qquad k=0,\ldots,r-1,\qquad z=2q-1,
$$

where $\phi_k$ are normalized Jacobi polynomials with parameters $(1,1)$,
orthonormal for weight $3(1-z^2)/4$ on $[-1,1]$. The envelope makes interior
basis functions vanish at the count endpoints. The full representation
has $r+2$ coordinates; the segregating representation has $r$.

Write the one-generation expectation approximation as $P\approx DE$, with
$E$ mapping values to coefficients and $D$ evaluating at the offspring
probabilities after the full frequency map. One-generation operations apply
these external maps directly. For Krylov evolution, $H=ED$ is a reduced square
matrix. In exact coefficient arithmetic,

$$
\widetilde P^{\,g}=DH^{g-1}E,\qquad g\ge1.
$$

The [analysis note](https://github.com/mancusolab/drifft/blob/main/notes/wf_jacobi_bounds.md)
defines the ideal projection and distinguishes it from sampled coefficients
and padded matrix assembly.

## Computation

Gauss–Jacobi quadrature and shifted binomial samples construct the external
coefficient transform. The implementation uses $Q=r+32$ quadrature nodes,
truncated and normalized binomial stencils, and reflection to store only half
of the quadrature geometry. Reflection remains valid for asymmetric inputs.

For supported frequency maps, the reduced matrix is assembled through rational
functions of a symmetric tridiagonal Hahn coordinate matrix. A leading
$R\times R$ block supplies padding; banded solves and a polynomial recurrence
produce the retained matrix and endpoint coupling. General diploid selection
uses a quadratic rational map and banded Cholesky solves; symmetric
underdominance retains its specialized even/odd solves when used without
mutation. Recurrent mutation can move mass away from zero or fixation, so the
full $r+2$ matrix includes their coupling to the interior. No dense population
transition is formed.

Forward mass is evaluated in the opposite order, using the transposed external
maps. For Krylov evolution, the source mass is encoded once, the method advances
its reduced state, and the output is decoded once.
[Krylov methods](jacobi_krylov.md) describe the power approximation and constant
immigration. Sequential repeats the one-generation count-space transition,
including the external transforms at each generation.

## Cost and reuse

Geometry depends on population size, rank, and numerical settings. It is
reusable across supported selection and mutation parameters within the planned
bounds. The reduced matrix depends on both selection and mutation.
RationalKrylov factors its shifted matrix once per power action and reuses that
factorization across basis vectors. The shift also depends on interval duration.

| Stage | Cost or storage |
| --- | --- |
| Reduced assembly | $O(rR)$ arithmetic |
| Reduced matrix storage and matvec | $O(r^2)$ |
| Dense LU for rational Krylov | $O(r^3)$ factorization; $O(r^2)$ per solve |
| Source-mass encoding | $O(Mr)$ with the streamed polynomial recurrence |
| Decoding | $O(Qr+QW+M)$ for stencil width $W$ |
| One-generation expectation at $N$ probabilities | $O(Qr+QW+M+Nr)$ |
| Stored quadrature geometry | $O(Qr+QW+M)$, with symmetry reducing constants |

With Krylov, encoding and decoding are paid at the interval boundaries rather
than for every reduced step. A reduced matvec is dense, not an FFT. Benefits
depend on how much repeated population-space work the reduction avoids and how well setup is
amortized. Finite-tolerance rank bounds also contain accuracy and population-size
factors; do not interpret $r$ as an unconditional constant times $\sqrt M$.

## Accuracy and controls

| Control | Role |
| --- | --- |
| `rank` | Retained interior polynomial modes |
| `one_step_tolerance` | Plans ideal truncation and assembly padding separately |
| `max_strength` | Static absolute bound on both diploid coefficients, used to plan padding |

The `max_strength` bound and supported-map restrictions apply to reduced Krylov
assembly. One-generation operations and Sequential evaluate the frequency map
directly and accept any map producing valid probabilities.

The error budget includes polynomial truncation, quadrature, binomial stencil
truncation, padded assembly, and arithmetic. The default rank certificate covers
ideal one-step projection; it is not a total implemented-error guarantee.
The quadrature order and stencil policy are implementation choices, not separate
public tolerances. Increase rank and check against a small dense reference or
an independently tightened FFT/NUFFT calculation.

The evolution method introduces its own error. A converged Krylov action does
not certify Jacobi representation accuracy. Validate the final statistic and
parameter derivatives independently, as described in the
[accuracy guide](../guide/numerics.md).

## Differentiation and adjoints

Selection, mutation rates, initial mass, and immigration are dynamic parameters.
Rank, padding, quadrature nodes, and population size are fixed geometry. JAX
differentiates mapped evaluation and rational assembly; the method also
differentiates its power action.

The encoder has explicit JVP and transpose rules. Polynomial moments and
their derivatives use streaming recurrences, keeping reverse-mode working
storage proportional to population size plus rank at fixed derivative order.
The rules differentiate the same polynomial approximation, including its
endpoint terms. Evolution methods supply separate power-sensitivity rules;
see [automatic differentiation](autodiff.md).

The external maps have defined transposes, and the full-state construction
preserves mass. Endpoint rows are absorbing when the frequency map fixes zero
and one; recurrent mutation generally removes that property. Segregating
evolution still discards endpoint mass each generation. The approximation need
not preserve positivity. Sequential provides a matched expectation
transpose for demographic evolution. An input-dependent Krylov approximation is
not automatically a fixed linear map with a matched expectation transpose;
Krylov demographic evolution exposes forward propagation only.

## Use in drifft

Use Jacobi in `Transition` for one-generation operations and expected-visits
solves, or as `backend=Jacobi` in `DemographicTransition` with any epoch method.
Sequential repeats the same one-generation approximation; PolynomialKrylov and
RationalKrylov use the reduced assembly. The
[Jacobi API](../api/backends.md#drifft.Jacobi) lists its supported maps and parameter
bounds. Size-changing steps use Jacobi by default; an explicit FFT or NUFFT
`boundary_backend` is also supported.

See [choosing algorithms](../guide/backends.md) for a complete example,
[direct epoch operations](epochs.md) for custom orchestration, and the
[Jacobi API](../api/backends.md#drifft.Jacobi) for constructor contracts.
