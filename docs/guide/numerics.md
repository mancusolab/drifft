# NUFFT accuracy and derivatives

NUFFT approximates binomial expectations with a polynomial, then evaluates it
at the offspring probabilities. Population size and accuracy settings determine
the polynomial degree. The same prepared backend can be reused as selection
parameters or input distributions change.

## Accuracy controls

| Option | Default | Meaning |
| --- | --- | --- |
| `accuracy` | `1e-10` | Nominal value-error budget; `[1e-13, .1)` |
| `tail_tol` | allocated automatically | Optional stricter omitted binomial mass per node |
| `nufft_tol` | `min(1e-13, accuracy/4)` | Requested FINUFFT tolerance; at least `1e-15` |
| `batch_size` | `32` | Sampling nodes processed together |
| `derivative_accuracy` | `False` | Increase degree by 50%, capped at exact degree; heuristic, not a derivative certificate |
| `dtype` | `jnp.float64` | Only float64; enable x64 before planning |

The budget is split between polynomial approximation, binomial sampling,
NUFFT evaluation, and roundoff. Tightening it increases the prepared degree
and the work per application.

!!! warning "Accuracy budgets are not error certificates"

    NUFFT tolerance and the roundoff allocation require empirical validation.
    Value-error bounds do not certify derivative accuracy; check gradients
    separately for the intended workload.

## Practical accuracy checks

Compare results at tighter `accuracy` settings and, for small populations,
against a dense binomial reference. Check the final quantity of interest and
its gradients separately. A small one-generation error can accumulate over
many generations or be amplified by an expected-visits solve.

Sampling weights are normalized within each retained binomial stencil. This is
part of constructing the approximation; transition outputs are not normalized.
Endpoint values are exact, but polynomial interpolation can introduce tiny
negative coefficients elsewhere.

## Differentiation and memory

Selection derivatives are evaluated as polynomials directly in the offspring
probability, so they remain finite at zero and one. Differentiate frequency-map
parameters or input vectors while keeping backend geometry fixed. Derivatives
of the backend's sampling nodes are unsupported.

NUFFT regenerates binomial weights in batches instead of retaining a weight
table. `batch_size` controls this temporary storage. FINUFFT also allocates
working memory outside JAX; XLA memory reports do not include it.

## Polynomial representation

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

## Derivative formula

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
