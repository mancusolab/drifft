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
y_r=a_r\sum_j e^{-\Delta_{\mathrm{logit}}(r-(j-j_0))^2/2}\,b_jf_j.
$$

Here $j_0$ is a reference count, $\Delta_{\mathrm{logit}}$ is the log-odds spacing, and $a_r,b_j$
are scaling weights. After scaling $f_j$ by $b_j$, the kernel depends only on
$r-(j-j_0)$. FFTs turn this convolution into pointwise multiplication in Fourier
space; an inverse FFT and scaling by $a_r$ recover the result. Zero-padding
prevents circular wraparound, and no dense block is constructed.
Local polynomial interpolation evaluates
the result at the required offspring probabilities. Near the endpoints,
direct binomial sums replace the log-odds interpolation.

## Cost and reuse

Preparation is reusable when population sizes and numerical settings stay fixed.
The cost of applying a transition includes both FFTs and interpolation.

For local blocks with padded FFT lengths $n_b$, the convolution work scales
with $\sum_b n_b\log n_b$. The complete operation also includes diagonal
scaling, interpolation at $n_{\mathrm{eval}}$ probabilities, and direct endpoint-region sums.
Their costs depend on the retained windows and interpolation stencil. There
is no sublinear-in-output guarantee from the FFT length alone.

Preparation caches binomial geometry, scaling factors, and kernel FFTs. They
are reused across vectors and selection parameters at fixed sizes and settings.
The transforms consume population-space values on each application; no dense
population transition is stored.

## Accuracy and controls

The Gaussian-shaped kernel above is a factorization of the binomial kernel,
not a normal approximation to the biological model. Errors arise from omitted
binomial tails, interpolation, and floating-point arithmetic.

| Control | Role |
| --- | --- |
| `tail_tol` | Omitted probability mass in retained binomial windows |
| `interp_order` | Local polynomial interpolation stencil size |
| `min_mode_count` | Extent of direct evaluation near endpoints |
| `block_spacing` | Separation of local grids |
| `balance` | Recenter diagonal scaling factors |

A tail tolerance does not bound interpolation or arithmetic error. For small
populations, the backend uses direct sums throughout. Check final results under
tighter settings and against dense binomial rows where feasible.

## Differentiation and adjoints

Forward propagation uses the transpose of the same expectation approximation,
so the two operations remain paired. Parameter derivatives follow the changing
interpolation and scaling, with geometry held fixed.

At the endpoints, direct binomial sums use the polynomial derivative identity

$$
\frac{\mathrm d B_{M,j}(q)}{\mathrm dq}
=M[B_{M-1,j-1}(q)-B_{M-1,j}(q)],
$$

where out-of-range indices contribute zero. This avoids singular expressions
at $q=0,1$. Validate gradients near interpolation-region changes; the
[autodiff chapter](autodiff.md) describes differentiation through evolution.
