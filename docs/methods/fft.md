# Local FFT convolution

`FFT` represents one-generation binomial expectations using local convolutions.
It operates on population-sized vectors; it does not maintain a reduced state
between generations.

## Problem and representation

Let $M$ be the destination gene-copy count and $N$ the number of source rows.
The input has $M+1$ values and the expectation output has $N$ entries.
The FFT backend computes binomial expectations through local log-odds grids.
For destination values $v_j$, it approximates

$$
B_M[v](q)=\sum_{j=0}^M\binom Mj q^j(1-q)^{M-j}v_j.
$$

## Computation

Within a retained local grid, the binomial kernel has a diagonal–Toeplitz–diagonal
factorization. The action takes the form

$$
y_r=a_r\sum_j e^{-h(r-(j-j_0))^2/2}\,b_jv_j.
$$

Here $j_0$ is a reference count, $h$ is the log-odds spacing, and $a_r,b_j$
are scaling weights. The sum is a convolution: zero-padded FFTs evaluate it
without constructing a dense block. Local polynomial interpolation evaluates
the result at the required offspring probabilities. Near the endpoints,
direct binomial sums replace the log-odds interpolation.

## Cost and reuse

For local blocks with padded FFT lengths $n_b$, the convolution work scales
with $\sum_b n_b\log n_b$. The complete operation also includes diagonal
scaling, interpolation at $N$ probabilities, and direct endpoint-region sums.
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

The expectation operation is linear in the input values. `Transition.advance`
uses its algebraic transpose, so propagation and expectations share the same
approximation. JAX differentiates the local interpolation and scaling as the
frequency map changes; the direct binomial path has polynomial endpoint
derivative rules. Validate parameter gradients, including near changes of
interpolation region. Geometry construction is outside the differentiated path.

## Use in drifft

Construct `FFT` outside JIT, then bind it to any supported frequency map with
`Transition`. It supports rectangular population-size changes, expected visits,
and demographic evolution with Sequential or either Krylov method. Each
Sequential generation still applies the population-space transition.

See [choosing algorithms](../guide/backends.md) for practical recommendations,
[accuracy](../guide/numerics.md) for validation, and the
[FFT API](../api/backends.md#drifft.FFT) for constructor defaults.
