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

With `Sequential`, JAX differentiates the repeated backend applications.
With either Krylov method, it differentiates the numerical approximation,
including basis construction, projection, the reduced matrix power, and
reconstruction. The basis depends on both the transition parameters and the
incoming state. That state can carry parameter dependence from earlier
demographic intervals. Jacobi's parameter-dependent encoding and reduced
operator are also part of the differentiated calculation.

drifft does not currently implement a custom JVP for an entire epoch or
`DemographicTransition`. In particular, it does not construct the block
operator above and apply Krylov to its power. Such an algorithm would
approximate the underlying model's sensitivity directly; at finite basis
dimension, that sensitivity need not equal the derivative of the current
Krylov approximation.

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
This is implicit differentiation of an inner linear solve, while the
surrounding Krylov basis construction remains differentiated explicitly.

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

Automatic differentiation follows the implemented numerical calculation.
Agreement with its finite differences does not establish accuracy for the
exact Wright–Fisher model. Increase backend accuracy and Krylov dimension
separately, and compare gradients with a dense reference where feasible.
Forward convergence diagnostics do not certify gradient accuracy.

Krylov derivatives at zero seeds or changes in Arnoldi breakdown are not
guaranteed. The accumulated recurrence-defect diagnostic is excluded from
differentiation; the propagated distribution remains differentiable. A failed
tangent or transpose solve can raise even when the primal call uses
`throw=False`. Measure derivative runtime and memory separately from forward
evaluation.

See [accuracy and differentiation](../guide/numerics.md) for validation
controls and [Krylov power methods](jacobi_krylov.md) for the forward algorithm.

## References

- [JAX automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html).
- [JAX custom derivative rules](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html).
- [Lineax linear solves](https://docs.kidger.site/lineax/api/linear_solve/).
