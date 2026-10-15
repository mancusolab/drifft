# Choosing a backend

Both backends implement `AbstractTransitionBackend`, an abstract Equinox module.
`Transition` owns selection, source size, propagation, and solver integration;
backends own selection-independent geometry and the expectation approximation.
Choose a backend explicitly and reuse it across selection strengths.

`FFT` uses local log-odds blocks, cached Fourier coefficients, and interpolation.
It favors repeated applications at a fixed size, at the cost of larger geometry
and XLA temporary storage. Its constructor accepts `tail_tol`, `interp_order`,
`min_mode_count`, `block_spacing`, and `balance`. Host NumPy/SciPy preparation
uses binomial quantiles and extended precision for conditioned Fourier weights;
application and differentiation use JAX arrays.

!!! warning "FFT interpolation error"

    Tail tolerance excludes interpolation and roundoff error. Interpolation can
    introduce small negative coefficients, and boundaries can limit gradient
    smoothness. Check gradients for the intended inference workload.

`NUFFT` stores a small node vector and regenerates sampling weights, trading
extra work per application for lower stored geometry and XLA temporary storage.
Its polynomial derivative rules remain finite at probability endpoints.
These custom operations use experimental `jax.experimental.hijax.HiPrim` to
preserve explicit JVP, transpose, and batching behavior under JAX transforms.
Hijax does not itself make gradients faster or reduce demographic recompilation:
changing population shapes can require new compilations for either backend.


## Local FFT operator

Write the binomial expectation as

$$
B_M[v](q)=\sum_{j=0}^{M}\binom{M}{j}q^j(1-q)^{M-j}v_j,
$$

where $M$ is the destination gene-copy count. For one interior block, choose
an integer reference count $j_0$, let $p_0=j_0/M$, and define

$$
\eta_0=\log\frac{p_0}{1-p_0},\qquad
h=\frac{1}{Mp_0(1-p_0)},\qquad
q_r=\frac{1}{1+\exp[-(\eta_0+hr)]}.
$$

At the integer grid nodes $r$, the binomial weights admit the factorization

$$
\binom{M}{j}q_r^j(1-q_r)^{M-j}
=a_r\,e^{-h[r-(j-j_0)]^2/2}\,b_j,
$$

with

$$
\begin{aligned}
a_r&=\frac{\exp(hrj_0+hr^2/2)}{(1-p_0+p_0e^{hr})^M},\\
b_j&=\binom{M}{j}p_0^j(1-p_0)^{M-j}
       \exp\!\left[\frac{h}{2}(j-j_0)^2\right].
\end{aligned}
$$

The Gaussian-shaped factor is an algebraic factorization of the binomial
kernel, not a normal approximation. The implementation redistributes a
constant scale between $a$ and $b$ for conditioning; their product is unchanged.
For retained columns $j$, define $w_{j-j_0}=b_jv_j$ and
$g_d=\exp(-hd^2/2)$. Each block evaluates

$$
y=a\odot\mathcal F^{-1}\!\left[\mathcal F(g)\odot\mathcal F(w)\right],
$$

with zero-padding and extraction of the required convolution entries.
Local Lagrange interpolation then evaluates $y$ at
$r(q)=[\log(q/(1-q))-\eta_0]/h$. Truncated input windows, interpolation,
and floating-point evaluation introduce approximation error. Boundary rows
use direct truncated binomial sums, with exact endpoint values.

## Chebyshev/NUFFT operator

NUFFT represents the same expectation by a global Chebyshev polynomial instead
of local log-odds interpolants. Sampling, coefficient conversion, and evaluation
are each linear in $v$. For either backend, `Transition.advance` applies the
transpose of the implemented expectation operator. See the
[NUFFT numerical guide](numerics.md) for the sampling nodes, cosine coefficients,
and derivative rules.
