# Local FFT convolution

[`FFT`](../api/backends.md#drifft.FFT) represents one-generation binomial expectations using local convolutions.
It operates on population-sized vectors; it does not maintain a reduced state
between generations.

Configuration: [API reference](../api/backends.md#drifft.FFT).

## Problem and representation

A single transition requires a binomial expectation at each source frequency.
FFT accelerates these expectations without storing the population transition matrix.

Let $M$ be the destination gene-copy count and $n_{\mathrm{eval}}$ the number of source rows.
The input has $M+1$ values and the expectation output has $n_{\mathrm{eval}}$ entries.
The FFT backend computes binomial expectations through local log-odds grids.
For destination values $f_j$, it approximates

$$
B_M[f](q)=\sum_{j=0}^M\binom Mj q^j(1-q)^{M-j}f_j.
$$

## Computation

Directly multiplying an $n$-by-$n$ binomial block by a vector costs $O(n^2)$:
each output requires a separate weighted sum. Factoring the kernel on a local
log-odds grid exposes a Toeplitz matrix, whose entries depend only on the index
difference. Its matrix–vector product is a convolution, which FFTs evaluate in
$O(n\log n)$ time. Diagonal scalings before and after the convolution cost only
$O(n)$, so they preserve this saving.

Within a retained local grid, this diagonal–Toeplitz–diagonal factorization gives

$$
y_t=a_t\sum_j e^{-h(kt-(j-j_0))^2/2}\,b_jf_j.
$$

Here $j_0$ is a reference count, $h$ is the fine log-odds spacing, $k$ is an integer
stride, and $a_t,b_j$ are scaling weights. After scaling $f_j$ by $b_j$, the kernel
depends only on the difference between the fine output index $kt$ and $j-j_0$.
FFTs turn this convolution into pointwise multiplication in Fourier
space; an inverse FFT and scaling by $a_t$ recover the result. Padding keeps circular wraparound outside the retained output indices.
Local polynomial interpolation evaluates
the result at the required offspring probabilities. Near the endpoints,
direct binomial sums replace the log-odds interpolation.

The two grids have separate roles. Convolution starts at
$h=1/(4Mp(1-p))$, where $p=j_0/M$, to control diagonal-factor growth.
Interpolation uses $H=kh$, selecting a power-of-two stride from the value and
gradient bounds. Only these coarse nodes are retained. For $L$ input values and a fine-grid output span of $R$ indices, the
kernel has $K=R+L-1$ entries. At a transform length of at least $K$, any
wrapped tail lands before index $L-1$. The coarse gather reads indices from
$L-1$ through $K-1$.

The input factors use outward `log1p` recurrences from the reference count.
The output factors use a centered binomial cumulant: a series resolves its
small quadratic terms, and log-space evaluation handles large shifts without
overflow. On the final retained count window,
the builder checks an exact-rational lower bound on binomial log curvature.
If necessary, it halves $h$ and doubles $k$, preserving $H$ and all interpolation
nodes while making the input factor strictly log-concave. By default it then
normalizes the input factor to maximum one, with the opposite scaling in the
output factor. These steps improve conditioning and exponent range; they do
not provide a complete floating-point error certificate.

## Cost and reuse

Preparation is reusable when population sizes and numerical settings stay fixed.
The cost of applying a transition includes both FFTs and interpolation.

For local blocks with padded FFT lengths $n_b$, the convolution work scales
with $\sum_b n_b\log n_b$. The complete operation also includes diagonal
scaling, interpolation at $n_{\mathrm{eval}}$ probabilities, and direct endpoint-region sums.
Their costs depend on the retained windows and interpolation stencil. There
is no sublinear-in-output guarantee from the FFT length alone.

Preparation builds binomial geometry, scaling factors, and kernel FFTs in
NumPy/SciPy, packs them into buckets, then creates the cached JAX arrays. They
are reused across vectors and selection parameters at fixed sizes and settings.
The transforms consume population-space values on each application; no dense
population transition is stored.

## Accuracy and controls

The Gaussian-shaped kernel above is a factorization of the binomial kernel,
not a normal approximation to the biological model. Errors arise from omitted
binomial tails, interpolation, and floating-point arithmetic.

| Control | Role |
| --- | --- |
| `atol`, `grad_atol` | Joint value and branchwise parameter-derivative targets |
| `tail_tol` | Omitted probability mass in retained binomial windows |
| `interp_order` | Optional explicit stencil size; overrides automatic order selection |
| `min_mode_count` | Initial extent of direct evaluation; automatic planning may enlarge it |
| `block_spacing` | Separation of local grids |
| `balance` | Normalize the input factors, compensating in the output factors |

A tail tolerance does not bound interpolation or arithmetic error. For small
populations, the backend uses direct sums throughout. Check final results under
tighter settings and against dense binomial rows where feasible.

`FFT(pop_size, atol=...)` plans interpolation and sampling on the host, with
one quarter of `atol` allocated to each. With `interp_order=None` (the default),
it tries orders 12, 14, and 16, starting the direct region at 32 expected copies.
Within each block it tests interpolation strides 1, 2, 4, up to 64 and retains
the coarsest passing stride before the first failed candidate. Each candidate's
full interpolation halo enters the bounds. If no order meets the budgets, it doubles
the boundary cutoff and retries, up to direct-only evaluation. This finite
search does not assume that higher order always improves the bound.
An explicit even `interp_order` from 2 through 16 fixes the requested
order and may produce an interpolation bound above the target, analogous
to an explicit Jacobi rank. If `grad_atol` is supplied, an explicit order
exceeding its interpolation-gradient budget raises `ValueError`.

Preparation stores `interpolation_bound` and `sampling_bound`.
`backend.error_report` reads these scalars without recomputing geometry or
transferring arrays to the host, using the same report type as NUFFT and Jacobi.
There is no separate diagnostic API.

The interpolation calculation bounds binomial derivative energies using exact
rational Bernstein polynomial enclosures. Sampling uses binomial tail
probabilities evaluated by SciPy, with a uniform factor of two covering the
signed interpolation weights for all supported orders. The node tail target
is at most `atol / 8`; boundary and node windows are enlarged if their computed
tail bounds exceed that target. Bound evaluation and tail probabilities still
have floating-point error; they are not an interval certificate for the entire
calculation.

The analytic bounds exclude errors from coordinate arithmetic, diagonal-factor
construction, cached kernels, and FFT application. `error_report` therefore
leaves `roundoff` and the total bound unknown. Small boundary cutoffs or wide
explicit blocks can still amplify arithmetic error even when every stored
factor is finite and the input factor is log-concave. Increasing order or
tightening tails does not necessarily improve numerical accuracy.

## Differentiation and adjoints

Forward propagation uses the transpose of the same expectation approximation,
so the two operations remain paired. Parameter derivatives follow the changing
interpolation and scaling, with geometry held fixed.

Within-block stencil switches have zero value jump but can have derivative
kinks. Direct/interior, block, and direct-only reflection interfaces can have
value jumps. Their size is bounded by twice the sum of the reported analytic
value contributions, but this does not establish global Sobolev regularity.
`grad_atol` promises branchwise analytic derivative accuracy; it does not bound
sensitivities across a value jump or imply a classical derivative at every
switch. `error_report.grad_coordinate` explicitly records this scope.

Supply `grad_atol` together with an `angular_sensitivity` bound on
$|\partial_s q|/\sqrt{q(1-q)}$ over the parameter domain, using the same
parameter-coordinate convention as NUFFT and Jacobi. Construction selects one
geometry satisfying both value and gradient allocations. There are no runtime
accuracy checks. Zero angular sensitivity preserves the value-only plan and
has zero analytic parameter-derivative error.

The derivative interpolation bound uses the differentiated divided-difference
estimate and the same binomial energy polynomials, at orders $m$ and $m+1$.
A uniform bound of four on the sum of absolute derivatives of interpolation
weights controls node-tail amplification. Direct-boundary derivative tails
use the binomial score identity and an analytic maximum over the boundary
interval. Each gradient contribution receives `grad_atol / 4`.
Preparation widens tail windows or increases order/cutoff as needed; numerical
planning failures raise `ValueError`. These targets exclude arithmetic error,
just as the value targets do. Absorbing endpoints must be parameter independent.

`interpolation_grad_bound` and `sampling_grad_bound` store parameter-derivative
bounds when requested, otherwise `None`. The shared report reads these cached
scalars; its total gradient bound remains `None` because roundoff is unbounded.

At the endpoints, direct binomial sums use the polynomial derivative identity

$$
\frac{\mathrm d B_{M,j}(q)}{\mathrm dq}
=M[B_{M-1,j-1}(q)-B_{M-1,j}(q)],
$$

where out-of-range indices contribute zero. This avoids singular expressions
at $q=0,1$. Validate gradients near interpolation-region changes; the
[autodiff chapter](autodiff.md) describes differentiation through evolution.
