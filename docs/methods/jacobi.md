# Jacobi reduction

`Jacobi` represents constant-size evolution in polynomial coefficients. Its
reduced state is used by an evolution method; it is a different approximation
from the Chebyshev interpolation used by `NUFFT`.

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
probabilities after the full frequency map. Then
$H=ED$ is a reduced square matrix. In exact coefficient arithmetic,

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
maps. The source mass is encoded once, the method advances its reduced state,
and the output is decoded once. [Krylov methods](jacobi_krylov.md) describe the
power approximation and constant immigration.

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
| Stored quadrature geometry | $O(Qr+QW+M)$, with symmetry reducing constants |

Encoding and decoding are paid at the interval boundaries rather than for every
reduced step. A reduced matvec is dense, not an FFT. Benefits depend on how much
repeated population-space work the reduction avoids and how well setup is
amortized. Finite-tolerance rank bounds also contain accuracy and population-size
factors; do not interpret $r$ as an unconditional constant times $\sqrt M$.

## Accuracy and controls

| Control | Role |
| --- | --- |
| `rank` | Retained interior polynomial modes |
| `one_step_tolerance` | Plans ideal truncation and assembly padding separately |
| `max_strength` | Static absolute bound on both diploid coefficients, used to plan padding |

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

The external maps have defined transposes, and the full-state construction
preserves mass. Endpoint rows are absorbing when the frequency map fixes zero
and one; recurrent mutation generally removes that property. Segregating
evolution still discards endpoint mass each generation. The approximation need
not preserve positivity. An input-dependent Krylov approximation is not
automatically a fixed linear map with a matched expectation transpose;
demographic Jacobi evolution exposes forward propagation only.

## Use in drifft

Use `backend=Jacobi` with `PolynomialKrylov` or `RationalKrylov` in
`DemographicTransition`. Supply an explicit FFT or NUFFT `boundary_backend` for
size-changing steps. Supported maps are neutral evolution, diploid selection
and its constrained classes, mutation, and one mutation composed with one
selection rule in either order. Both selection coefficients must have absolute
value at most `max_strength`; `Underdominance` and `Overdominance` additionally
require finite, nonnegative strength, including in mutation compositions.
Each mutation rate must be finite and in `[0, 1]`. Arbitrary frequency functions
and longer compositions are not supported. `Jacobi` is not a one-generation
`Transition` backend; use FFT or NUFFT for `Transition` and expected-visits solves.

See [choosing algorithms](../guide/backends.md) for a complete example,
[direct epoch operations](epochs.md) for custom orchestration, and the
[Jacobi API](../api/backends.md#drifft.Jacobi) for constructor contracts.
