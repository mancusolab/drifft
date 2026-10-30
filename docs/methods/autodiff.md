# Automatic differentiation and gradients

Let $T=T(\theta)$ be a forward transition operator, acting on a column
distribution $v_0$. For a fixed population size and frequency map, evolution
through $K$ generations gives

$$
v_K=T(T(\cdots T v_0))=T^K v_0.
$$

For a scalar parameter $\theta$ and a parameter-independent initial state,
the product rule gives

$$
\frac{\partial v_K}{\partial\theta}
=\sum_{j=0}^{K-1}
T^{K-1-j}\frac{\partial T}{\partial\theta}T^j v_0.
$$

Each term accounts for changing one generation's transition and propagating
that change to the final time. This is a finite-time sensitivity: it does not
require recurrent mutation, ergodicity, or a stationary distribution. If
$v_0$ depends on $\theta$, add $T^K\partial v_0/\partial\theta$.

## Forward sensitivities and demographic histories

Write $\dot v_t=\partial v_t/\partial\theta$ and
$\dot T=\partial T/\partial\theta$. The same derivative follows the recurrence

$$
v_{t+1}=Tv_t,\qquad
\dot v_{t+1}=T\dot v_t+\dot T v_t.
$$

Equivalently, the state and sensitivity evolve together:

$$
\begin{pmatrix}T&\dot T\\0&T\end{pmatrix}^{K}
\begin{pmatrix}\dot v_0\\v_0\end{pmatrix}
=\begin{pmatrix}\dot v_K\\v_K\end{pmatrix}.
$$

Across a demographic history, replace $T$ by the generation-specific operator
$T_t$. The recurrence still applies when a population-size change makes
$T_t$ rectangular. drifft holds the source size for all but the final
generation of each interval, then applies a boundary transition to the
destination size. Derivatives pass through both portions and into the next
interval. For immigration $v_{t+1}=T_t v_t+b_t$, the sensitivity also includes
$\dot b_t$.

Population sizes and integer generation counts are fixed configuration in
drifft. Differentiable inputs include selection coefficients, mutation rates,
initial distributions, and immigration. Differentiating a selection parameter
through a demographic history does not differentiate the history's sizes or
durations.

## JAX transformations

[`jax.jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html) computes
a Jacobian-vector product: the output change in a specified parameter
direction. This propagates a state and tangent, as in the recurrence above.
For several parameters, one JVP can evaluate their combined directional
effect without forming a full Jacobian.

[`jax.grad`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html) uses
reverse mode for a scalar output, such as a log likelihood. It propagates
sensitivities backward through the calculation and returns derivatives with
respect to the requested inputs. Reverse mode is usually preferable for a
scalar objective with many parameters, but can retain more intermediate state.
`jax.jit` compiles either calculation; compilation does not change which
parameters are differentiated.

The [selection-sensitivity example](../examples/selection_sensitivity.md)
shows a complete finite-time calculation with `jax.value_and_grad` and
finite-difference checks. Construct geometry outside the differentiated
function and replace dynamic parameter leaves inside it.

## What drifft differentiates

With `Sequential`, an operator-level primitive supplies a coupled state/tangent
recurrence and an explicit initial-state transpose using the transposed
operator. Operator tangents come from filtered JVPs of its matrix-vector
action, so dense and callable representations follow the same rule. Ordinary
AD still differentiates preparation and reconstruction, including immigration.
This boundary is internal; existing JAX and Equinox transformations use it
without an additional user option.
With either Krylov method, custom first-order rules approximate the sensitivity
of the prepared power action. They do not differentiate Arnoldi basis construction.
A coupled block action computes the JVP, using up to twice `dimension` basis
vectors for the doubled working state. The operator pullback uses two Krylov
spaces, each with up to `dimension` vectors, seeded by the initial state and
output cotangent. Their small projected matrices determine low-rank adjoint
factors; contractions with operator VJPs return cotangents in the original
operator PyTree without forming a population-sized matrix.

`DemographicTransition` composes these operations through ordinary AD. Jacobi's
parameter-dependent encoding and reduced operator, immigration, and population-size
boundaries remain in the differentiated calculation. The same method rules
apply to FFT, NUFFT, and Jacobi representations.

At finite basis dimension, these derivatives need not equal the derivative of
the executed primal Krylov approximation. They approximate the underlying
power's sensitivity. Forward and reverse approximations must be checked for
accuracy and adjoint consistency as the basis dimension increases.

Three functions have explicit custom JVP rules:

| Function | Purpose |
| --- | --- |
| Diploid selection | Analytic frequency and coefficient derivatives, factored to reduce overflow and cancellation; shared by the constrained diploid models |
| Multiplicative selection | Analytic frequency and fitness-ratio derivatives without squaring the mean fitness |
| FFT binomial weights | Polynomial derivatives with finite, correct values at offspring probabilities zero and one |

JAX uses these JVP rules for forward mode and constructs reverse-mode rules
from them. The binomial rule uses
$\partial B_{n,j}(q)/\partial q=n[B_{n-1,j-1}(q)-B_{n-1,j}(q)]$,
with out-of-range binomial indices interpreted as zero.

NUFFT also has explicit primitives for endpoint-safe polynomial evaluation
and its transpose, and for streaming binomial sampling. Jacobi encoding uses
a polynomial-moment primitive with a streaming evaluation transpose. Its JVP
differentiates both the evaluation points and input mass using polynomial
derivative recurrences. These rules support higher derivatives and avoid
retaining a population-sized vector for every polynomial degree in reverse
mode. For fixed derivative order, encoder working storage is $O(M+r)$ for
$M$ gene copies and rank $r$; arithmetic remains $O(Mr)$. Separate recurrence
passes can increase execution time compared with retaining intermediates.

## Implicit derivatives of linear solves

Rational Krylov constructs its basis using shifted systems

$$
B x=b,\qquad B=I+\tau(I-A),\quad \tau>0,
$$

where $A$ is the prepared working-state operator. Jacobi uses dense LU solves;
FFT and NUFFT use GMRES. Through Lineax, these solves differentiate the
defining equation:

$$
B\dot x=\dot b-\dot B x.
$$

The derivative requires another solve, rather than differentiation through
the factorization or GMRES iterations. Reverse mode uses a transposed system.
Expected-visits derivatives use this implicit solve rule. Krylov power
derivatives instead apply their own mathematical sensitivity rule; they do not
differentiate basis construction. Dense rational block sensitivities reuse one
factorization of the original shifted operator for both diagonal blocks.

Invertibility of $B$ is the relevant condition. Absorption does not itself
cause singularity: an eigenvalue $1$ of $A$ becomes eigenvalue $1$ of $B$.
For a stochastic $A$, the positive shift gives an invertible system; reduced
numerical operators still require numerical validation.

Expected-visits calculations also use Lineax's implicit solve derivatives.
Their transient system is invertible when eventual absorption occurs with
probability one. This differs from differentiating a stationary distribution:
without recurrent mutation, the full chain has multiple stationary
distributions supported on its absorbing boundaries.

## Accuracy and limitations

Sequential differentiates its repeated numerical transition. Krylov methods
approximate the sensitivity of the prepared power, so finite differences of an
underresolved primal approximation are not sufficient validation. Increase
backend accuracy and Krylov dimension separately; use dense references and
JVP/VJP adjoint-pairing checks where feasible.

Zero seeds and exact invariant-subspace breakdown are supported by the new
first-order Krylov rules. The sensitivity and operator pullback have separate
convergence estimates. The coupled action checks a combined error estimate
against both the state and sensitivity tolerances. The operator pullback combines a nested-space Frobenius difference with
recurrence-defect estimates along both intermediate trajectories. This avoids
accepting two vanishing projected pullbacks when propagation is unresolved.
The defect estimate uses unit amplification and is heuristic for general
nonnormal operators. These diagnostics are not certificates.
A failed sensitivity or pullback raises even with `throw=False`; there is no
automatic fallback. A failing member under `vmap` can fail the batch.

Higher-order differentiation through the Krylov Fréchet rule is not currently
implemented and raises explicitly. Use Sequential for such
calculations. Sequential and the backend-level polynomial derivative rules
retain their higher-order support.

Error estimates and status fields are numerical diagnostics rather than
differentiation targets. Measure derivative runtime and memory separately from
forward evaluation. Neither primal nor derivative convergence establishes total
accuracy for the exact Wright–Fisher model.

See [accuracy and differentiation](../guide/numerics.md) for validation
controls and [Krylov power methods](jacobi_krylov.md) for the forward algorithm.

## References

- [JAX automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html).
- [JAX custom derivative rules](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html).
- [Lineax linear solves](https://docs.kidger.site/lineax/api/linear_solve/).
- [Kandolf et al., low-rank Fréchet derivative approximations](https://arxiv.org/abs/2008.12926).
