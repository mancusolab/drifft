# Automatic differentiation and gradients

Here we present the mathematical aspects of differentiating Wright–Fisher
evolution and how drifft computes these derivatives. We distinguish the
one-generation transition, the power used to evolve through an interval, and
the encoding and reconstruction around that power. Each derivative identity
below is paired with the operation that implements it.

!!! note "Advanced: matrix-power differentiation"

    The derivations introduce Fréchet derivatives and use Krylov projection to
    compute sensitivities without forming a matrix-sized operator cotangent.
    Read the [Krylov chapter](jacobi_krylov.md) for the projection construction,
    or start with the [selection-sensitivity example](../examples/selection_sensitivity.md)
    and [derivative support](#accuracy-and-limitations) for practical use.

## Problem and representation

Evolution derivatives combine changes in the transition operator with changes
in the input mass. Separating those contributions gives rules that work for
both population and reduced coordinates.

Let $\theta$ collect the differentiable model parameters and $v$ be a column
vector of population mass, following the [shared notation](../guide/notation.md).
The [model chapter](../guide/model.md) defines the exact expectation matrix $P$.
For $g$ generations at fixed size and parameters, the exact target is
$v_g=(P^T)^g v_0$.

Write $T=\widetilde P^T$ for the represented forward operator. The calculations
below use

$$
v_{t+1}=T(\theta)v_t,
$$

and differentiate this numerical representation. Its approximation error is
separate from the error of a method used to compute the power $T^g$.

| Mathematical operation | drifft operation |
| --- | --- |
| $\widetilde P f$: destination values to source expectations | `Transition.expect(values)` |
| $Tv$: source mass to destination mass | `Transition.advance(distribution)` |
| $T^g v$: evolution across a constant-size interval | An evolution method's `advance(...)` |
| Composition of intervals and size changes | `DemographicTransition.advance(...)` |

Selection coefficients, mutation rates, initial mass and immigration can be
differentiated. Population sizes, generation counts, ranks and method settings
are fixed configuration. Prepared backend geometry is held fixed while dynamic
model parameters change.

### The prepared power

Encoding and reconstruction surround the power calculation. Keeping them
explicit ensures their parameter dependence is included in the derivative.
An evolution method receives a working-state problem

$$
y=A^k z,\qquad v_{\mathrm{out}}=Ry+c.
$$

Here $A$ is the working-state operator, $z$ the encoded initial state, $k$ a
nonnegative integer exponent, $R$ a linear reconstruction map, and $c$ an
output offset. These need not have the same dimensions as the population state.

For an interval without immigration:

| Representation | $z$ | $A$ | $k$ | $R$ |
| --- | --- | --- | --- | --- |
| Population coordinates: Sequential with any backend; Krylov with FFT, NUFFT or Dense | $v_0$ | $T$ | $g$ | $I$ |
| Reduced coordinates: Jacobi with Krylov, $g\ge1$ | $D^T v_0$ | $H_{\mathrm J}^T$, where $H_{\mathrm J}=ED$ | $g-1$ | $E^T$ |

The Jacobi encoding includes one selected transition factor, hence $k=g-1$;
zero generations is an identity. The [Jacobi chapter](jacobi.md) derives this
representation and its approximation error.

## Forward sensitivities: the JVP

A JVP measures how the output changes along one parameter or state perturbation.
For a matrix power, that change is a sum over the steps at which the operator
can be perturbed.

A dot denotes a directional derivative along a supplied perturbation of
$\theta$. For $y=A^k z$, define the Fréchet derivative of the matrix power by

$$
\mathcal L_k(A,\Delta A)
=\sum_{j=0}^{k-1}A^{k-1-j}(\Delta A)A^j,
\qquad \mathcal L_0(A,\Delta A)=0.
$$

The joint operator-and-state derivative is

$$
\boxed{\dot y=\mathcal L_k(A,\dot A)z+A^k\dot z.}
$$

The first term changes the operator at each possible step; the second changes
the initial state. This is a finite-time derivative and requires neither a
stationary distribution nor ergodicity.

### Sequential: evolve the state and tangent together

Sequential evolution can carry a tangent alongside the state, applying the
product rule at each generation. Set $z_0=z$ and $\dot z_0=\dot z$. Differentiate each step:

$$
\begin{aligned}
z_{t+1}&=Az_t,\\
\dot z_{t+1}&=A\dot z_t+\dot A z_t.
\end{aligned}
$$

`Sequential` implements this coupled recurrence. The action $\dot A z_t$
comes from a filtered JVP of the operator's matrix-vector action, so the same
rule works for explicit matrices and callable operators.

### Krylov: approximate a coupled block power

Stacking the state and tangent turns the sensitivity into another power
problem, which either Krylov method can approximate. The recurrence becomes:

$$
\boxed{
\begin{pmatrix}A&\dot A\\0&A\end{pmatrix}^{k}
\begin{pmatrix}\dot z\\z\end{pmatrix}
=
\begin{pmatrix}\dot y\\y\end{pmatrix}.
}
$$

Both Krylov methods approximate this block action with their respective
projection algorithms. The doubled state can use up to twice `dimension`
basis vectors. For a pure state perturbation, $\dot A=0$, the derivative
reduces to another power action $A^k\dot z$.

## Reverse sensitivities: the VJP

Reverse mode propagates sensitivity from a scalar objective back to the
initial state and model parameters. It avoids a separate forward sensitivity
calculation for every parameter.

Let $\ell(y)$ be a scalar objective and let $\bar y=\nabla_y\ell$ be the
cotangent arriving at the power action. Define the forward and backward orbits

$$
z_j=A^j z,\qquad
\lambda_j=(A^T)^{k-j}\bar y,\qquad j=0,\ldots,k.
$$

Pairing the JVP with $\bar y$ gives the reverse sensitivities.

Thus the input cotangent and matrix cotangent are

$$
\boxed{\bar z=(A^T)^k\bar y,\qquad
G=\sum_{j=0}^{k-1}\lambda_{j+1}z_j^T,}
$$

where $\dot\ell=\bar z^T\dot z+\langle G,\dot A\rangle_F$ and
$\langle U,V\rangle_F=\operatorname{tr}(U^TV)$.
For $k=0$, $\bar z=\bar y$ and $G=0$.

### Compute operator cotangents without forming a large matrix

The matrix cotangent can be population-sized even when the final objective
is scalar. A factored approximation lets drifft contract it with operator
parameters without constructing that matrix.

!!! note "Advanced: low-rank adjoints"

    This derivation uses two Krylov projections and a block matrix power.
    The [accuracy contract](#accuracy-and-limitations) is sufficient if you
    only need to use and validate gradients.

Sequential obtains reverse mode by transposing its coupled derivative rule;
the initial-state transpose applies the transposed operator repeatedly.

Krylov approximates $G$ in two spaces: a basis $U$ for the backward orbit
under $A^T$, and a basis $V$ for the forward orbit under $A$. Each space uses
at most $m=$ `dimension` vectors. With

$$
H_{m,\mathrm L}=U^TA^TU,\quad H_{m,\mathrm R}=V^TAV,\quad
\alpha=U^T\bar y,\quad \beta=V^Tz,
$$

the projected matrix cotangent is

$$
G\approx UCV^T,\qquad
C=\sum_{j=0}^{k-1}H_{m,\mathrm L}^{k-1-j}\alpha\beta^T(H_{m,\mathrm R}^T)^j.
$$

The small matrix $C$ is the upper-right block of

$$
\begin{pmatrix}H_{m,\mathrm L}&\alpha\beta^T\\0&H_{m,\mathrm R}^T\end{pmatrix}^{k}.
$$

drifft retains the factors $F=UC$ and $V$. For operator parameters $\theta$,
it contracts their columns through VJPs of the operator action:

$$
\bar\theta_A\approx
\sum_i
\left(\frac{\partial\,[A(\theta)V_i]}{\partial\theta}\right)^T F_i.
$$

Here $F_i$ and $V_i$ are held fixed during each contraction. This returns
cotangents in the operator's original parameter PyTree without materializing
$G$ or a population-sized Jacobian.

## Encoding, immigration and demographic histories

Parameters can affect the transforms and source terms as well as the power.
The chain rule connects those contributions across an entire demographic history.
For
$v_{\mathrm{out}}=Ry+c$ and $y=A^k z$,

$$
\dot v_{\mathrm{out}}
=\dot R\,y
 +R\left[\mathcal L_k(A,\dot A)z+A^k\dot z\right]
 +\dot c.
$$

For example, Jacobi's encoded state $z=D^Tv_0$ contributes

$$
\dot z=\dot D^T v_0+D^T\dot v_0.
$$

Ordinary JAX differentiation handles preparation and reconstruction around
the custom power rule. Selection-dependent encoding, reduced assembly and
source terms therefore remain connected to model parameters.

### Immigration is an affine recurrence

Immigration adds mass after each generation. Augmenting the state with a
constant coordinate lets the same power machinery handle this affine update.
For $v_{t+1}=Tv_t+b$, the derivative is

$$
\dot v_{t+1}=T\dot v_t+\dot T v_t+\dot b.
$$

A constant source can be included in a linear power by augmenting the state:

$$
\begin{pmatrix}v_{t+1}\\1\end{pmatrix}
=
\begin{pmatrix}T&b\\0&1\end{pmatrix}
\begin{pmatrix}v_t\\1\end{pmatrix}.
$$

The prepared power uses this construction. In Jacobi coordinates, earlier
source events are encoded into the reduced recurrence and the last event is
added by the output offset $c$. The
[epoch operations page](../guide/epochs.md#mass-and-immigration) specifies the timing.

### Size changes compose through the chain rule

Population-size changes alter the state dimension but still compose through
the chain rule. For a generation-specific, possibly rectangular forward operator $T_t$,

$$
\dot v_{t+1}=T_t\dot v_t+\dot T_t v_t+\dot b_t.
$$

`DemographicTransition` applies the interval method for the first $g-1$
generations, then a boundary transition for the final generation. Ordinary AD
composes their derivatives in chronological order, and their pullbacks in
reverse order. Population sizes and durations remain fixed.

## How drifft connects these rules to JAX

Custom derivative rules let JAX use the power identities above without
tracing through the adaptive Krylov basis. All drifft-owned custom
differentiation uses `jax.experimental.hijax.HiPrim`.

HiJAX derives linearization and reverse mode from JVP rules; explicit
transposes handle linear inputs. The Krylov Fréchet primitive supplies its
operator pullback explicitly. Ordinary JAX differentiation connects these
rules to preparation, reconstruction, and the frequency map; Dense uses
ordinary JAX AD as an independent binomial reference.

!!! warning "Derivative of the power versus derivative of the algorithm"

    At finite basis dimension, drifft approximates $\mathrm d(A^k z)$.
    It does **not** differentiate the executed approximation
    $V(V^TAV)^kV^Tz$, whose basis depends on its inputs. These derivatives need
    not agree until the approximations are sufficiently accurate.

Use `jax.jvp` for a parameter direction and `jax.grad` or
`jax.value_and_grad` for a scalar objective. The
[selection-sensitivity example](../examples/selection_sensitivity.md) shows
parameter replacement and gradient validation. Backend chapters describe the
one-generation derivative rules.

## Implicit derivatives of linear solves

Linear solves appear in expected visits and rational Krylov construction.
Their derivatives follow from the defining equation, so they do not require
differentiating an iterative solver trajectory.

A solve $Bz=b$ has directional derivative and reverse solve

$$
\boxed{B\dot z=\dot b-\dot Bz,\qquad B^T\lambda=\bar z.}
$$

Consequently, $\bar b=\lambda$ and $\bar B=-\lambda z^T$.
Lineax differentiates this defining equation rather than the factorization or
iterative-solver trajectory.

| Use of a solve | Role in differentiation |
| --- | --- |
| Expected visits: $(I-Q^T)u=e_1$ | Lineax's implicit rule differentiates the occupation vector $u$ |
| Rational Krylov: $[I+\tau(I-A)]z=b$ | Builds the primal and sensitivity spaces; the outer power derivative uses the custom rules above |

These equations require invertible $B$. For rational Krylov, an eigenvalue
$1$ of $A$ becomes eigenvalue $1$ of $B=I+\tau(I-A)$; absorption alone does
not make this shifted system singular. Expected visits instead require a
transient interior kernel. Solve strategies are described in the
[Krylov chapter](jacobi_krylov.md#computation).

## Accuracy and limitations

A successful forward calculation does not establish derivative accuracy.
The derivative target and supported derivative orders depend on the evolution
method, and Krylov sensitivities have their own convergence checks.

| Calculation | Derivative target | Higher-order support |
| --- | --- | --- |
| Sequential | Repeated numerical transition, including its backend approximation | Supported through the recurrence and supported backend rules |
| PolynomialKrylov / RationalKrylov | Approximate sensitivity of the prepared power $A^k z$ | Higher-order differentiation through the Fréchet rule raises explicitly |

Zero seeds and invariant-subspace breakdown are supported by the first-order
Krylov rules when their convergence checks pass. Derivative checks are separate
from the primal checks:

- The coupled JVP's combined error estimate must satisfy both the state and
  sensitivity tolerances.
- The operator pullback combines a nested-space Frobenius difference with
  recurrence defects along both projected orbits. This detects unresolved
  propagation even when both projected pullbacks vanish.
- A failed sensitivity or pullback raises even with `throw=False`; a failing
  member under `vmap` can fail the batch. There is no automatic fallback.

Defect estimates use unit amplification and are heuristic for general nonnormal
operators. They do not certify total Wright–Fisher model accuracy. Error
estimates and status fields are diagnostics, not differentiation targets.

Validate by increasing backend accuracy and Krylov dimension separately,
comparing against small dense references, and checking JVP/VJP adjoint pairing.
Finite differences of an underresolved Krylov primal need not match its custom
derivative. Measure derivative time and memory separately from forward costs.
See [accuracy and differentiation](../guide/numerics.md) for the validation
procedure and [Krylov power methods](jacobi_krylov.md) for the primal algorithms.

## References

The JAX and Lineax references describe the differentiation interfaces;
Kandolf et al. develop the low-rank matrix-function derivative approach.

- [JAX automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html).
- [JAX custom derivative rules and HiJAX](https://docs.jax.dev/en/latest/301/custom-derivatives.html).
- [Lineax linear solves](https://docs.kidger.site/lineax/api/linear_solve/).
- [Kandolf et al., low-rank Fréchet derivative approximations](https://arxiv.org/abs/2008.12926).
