# Differentiating Wright–Fisher evolution

Here we derive the sensitivities of Wright–Fisher evolution and the numerical
rules drifft uses to compute them. The derivation proceeds from the first
variation of a transition to adjoints of matrix powers, their Krylov
approximations, and implicit derivatives of occupation equations.
Notation follows the [shared conventions](../guide/notation.md); practical
usage and validation are covered in the [user guide](../guide/numerics.md#differentiating-biological-parameters).

## Problem and assumptions

Differentiation acts on model parameters and states while population sizes,
generation counts, and numerical dimensions remain fixed. We distinguish the
exact biological operator from its numerical representation so that the target
of each derivative is explicit.

Let $\theta\in\mathbb R^{n_\theta}$ collect the $n_\theta$ differentiable inputs. The exact
one-generation expectation matrix is $P_t(\theta)$, and its numerical
representation is $\widetilde P_t(\theta)$. Both act on observables; their
transposes propagate population mass.
With a source vector $b_t$ added after each generation,

$$
v_{t+1}=\widetilde P_t^T  v_t+b_t,\qquad t=0,\ldots,g-1,
\qquad \ell=\ell(v_g,\theta).
$$

The matrices may be rectangular at population-size changes. All first-variation
identities below assume differentiability at the parameters considered. A dot
means differentiation along a direction $\dot\theta$, including any induced
changes in $v_0$ and $b_t$. Transposes and inner products are real.

Prepared grids, ranks, quadrature nodes, and interpolation settings are held
fixed. Selection-dependent operator entries and transforms are differentiated.
The identities are exact for the represented operators; this alone does not
establish accuracy relative to the exact Wright–Fisher process.

## First variation and matrix powers

The first variation separates perturbations already present in the state from
those introduced by the next transition. Iterating this identity gives the
sensitivity of a whole history without a stationarity or mixing assumption.

Expanding one step to first order,

$$
\begin{aligned}
(\widetilde P_t^T +\varepsilon\dot{\widetilde P}_t^{\,T})(v_t+\varepsilon\dot v_t)
 +b_t+\varepsilon\dot b_t
&=\widetilde P_t^T v_t+b_t\\
&\quad+\varepsilon(\widetilde P_t^T \dot v_t+\dot{\widetilde P}_t^{\,T}v_t+\dot b_t)
 +O(\varepsilon^2).
\end{aligned}
$$

Hence

$$
\boxed{\dot v_{t+1}=\widetilde P_t^T \dot v_t+\dot{\widetilde P}_t^{\,T}v_t+\dot b_t.}
$$

`Sequential` implements the corresponding coupled state/tangent recurrence.
The action $\dot{\widetilde P}_t^{\,T}v_t$ comes from differentiating the operator action with
its vector argument held fixed; no explicit derivative matrix is required.

### Constant-operator power

Within a constant-size interval, repeated application becomes a matrix power.
Its first variation sums the contributions from perturbing each factor in the
product; this is the derivative subsequently approximated by Krylov methods.

The operator being powered depends on the coordinates used for evolution.
In population coordinates it is $\widetilde P^T$. With Jacobi's factorization
$\widetilde P=DE$, evolution instead powers the smaller matrix
$H_{\mathrm J}^T=(ED)^T$ between encoding and reconstruction.

We use $A$ only for this **working-coordinate operator**, with the following
explicit correspondence for an interval without immigration:

| Coordinates | Operator $A$ | Input $z$ | Exponent $k$ | Population output |
| --- | --- | --- | --- | --- |
| Population | $\widetilde P^T$ | $v_0$ | $g$ | $v_g=y$ |
| Jacobi, $g\ge1$ | $H_{\mathrm J}^T$ | $D^Tv_0$ | $g-1$ | $v_g=E^Ty$ |

The common power calculation is $y=A^kz$. Thus $P$ always denotes the
population expectation matrix, whereas $A$ denotes the operator in the
coordinates actually evolved; it may even have a different dimension.
The [coordinate-transform derivation](#jacobi-coordinates) explains the Jacobi
row. The product rule for this common calculation gives

$$
\mathcal L_k(A,\Delta A)
:=\mathrm D(A^k)[\Delta A]
=\sum_{j=0}^{k-1}A^{k-1-j}(\Delta A)A^j,
\qquad \mathcal L_0(A,\Delta A)=0.
$$

This linear map in $\Delta A$ is the Fréchet derivative of the power function.
Including the input-state perturbation gives

$$
\boxed{\dot y=\mathcal L_k(A,\dot A)z+A^k\dot z.}
$$

For $k=0$, this reduces to $\dot y=\dot z$. No inverse of $A$ appears, so
singular transitions and absorbing states do not invalidate the identity.

### Coupled block representation

The tangent recurrence can itself be evaluated as a power action. This lets
drifft apply its evolution algorithms to the derivative without constructing
the full Fréchet derivative as a linear operator on matrices.

Define the coupled operator and its input and output states by

$$
A_{\mathrm{blk}}=\begin{pmatrix}A&\dot A\\0&A\end{pmatrix},\quad
z_{\mathrm{blk}}=\begin{pmatrix}\dot z\\z\end{pmatrix},\quad
y_{\mathrm{blk}}=\begin{pmatrix}\dot y\\y\end{pmatrix}.
$$

Block multiplication and induction give

$$
A_{\mathrm{blk}}^k=
\begin{pmatrix}A^k&\mathcal L_k(A,\dot A)\\0&A^k\end{pmatrix},
\qquad
\boxed{
A_{\mathrm{blk}}^k\begin{pmatrix}\dot z\\z\end{pmatrix}
=\begin{pmatrix}\dot y\\y\end{pmatrix}.}
$$

Indeed, the upper-right block at step $k+1$ is
$A\mathcal L_k(A,\dot A)+\dot A A^k=\mathcal L_{k+1}(A,\dot A)$.
For a pure state perturbation, $\dot A=0$, only the action $A^k\dot z$ is needed.

## Adjoint and parameter gradients

An adjoint contracts the first variation with a scalar objective, avoiding a
separate tangent calculation for every parameter. The same pairing yields both
the demographic-history gradient and the matrix cotangent used in the power rule.

Set $\bar v_g=\nabla_{v_g}\ell$ and define
$\bar v_t=\widetilde P_t\bar v_{t+1}$. Pairing the tangent recurrence with
$\bar v_{t+1}$ gives

$$
\bar v_{t+1}^T\dot v_{t+1}-\bar v_t^T\dot v_t
=\bar v_{t+1}^T(\dot{\widetilde P}_t^{\,T}v_t+\dot b_t).
$$

Summing telescopes, so

$$
\dot\ell
=\bar v_0^T\dot v_0
 +\sum_{t=0}^{g-1}\bar v_{t+1}^T(\dot{\widetilde P}_t^{\,T}v_t+\dot b_t)
 +\left.\partial_\theta\ell\right|_{v_g}\dot\theta.
$$

Consequently, for parameter $\theta_a$,

$$
\boxed{
\frac{\mathrm d\ell}{\mathrm d\theta_a}
=\bar v_0^T\frac{\partial v_0}{\partial\theta_a}
 +\sum_{t=0}^{g-1}\bar v_{t+1}^T
 \left(\frac{\partial\widetilde P_t^T}{\partial\theta_a}v_t
       +\frac{\partial b_t}{\partial\theta_a}\right)
 +\left.\frac{\partial\ell}{\partial\theta_a}\right|_{v_g}.}
$$

This also covers rectangular boundary transitions: the transpose maps each
adjoint back to the preceding state's dimension.

### Matrix cotangent of a power

For a constant operator, contributions from all steps can be collected into
one matrix cotangent. This expression is useful for deriving a factored
parameter pullback even when forming the matrix would be too expensive.

For $y=A^kz$ and incoming cotangent $\bar y$, define
$z_j=A^jz$ and $\bar z_j=(A^T)^{k-j}\bar y$. Thus $z_0=z$,
$z_k=y$, $\bar z_k=\bar y$, and $\bar z_0=\bar z$: the bars consistently
identify cotangents of the corresponding states. Then

$$
\bar y^T\dot y
=\bar z_0^T\dot z
 +\sum_{j=0}^{k-1}\bar z_{j+1}^T\dot A z_j.
$$

Each operator contribution satisfies
$\bar z_{j+1}^T\dot A z_j=\langle\bar z_{j+1}z_j^T,\dot A\rangle_F$,
where $\langle X,Y\rangle_F=\operatorname{tr}(X^TY)$. Therefore

$$
\boxed{\bar z=(A^T)^k\bar y,\qquad
\bar A=\sum_{j=0}^{k-1}(A^T)^{k-1-j}\bar y z^T(A^T)^j,}
\qquad
\dot\ell=\bar z^T\dot z+\langle \bar A,\dot A\rangle_F.
$$

Thus $\bar A$ is the cotangent of $A$. For $k=0$, $\bar A=0$ and $\bar z=\bar y$.
Sequential reverse mode follows by transposing its coupled derivative rule.

## Krylov sensitivity approximations

Krylov methods replace large power actions by projected ones. Forward and
reverse sensitivities use their own projection spaces, rather than derivatives
of the basis selected during the primal calculation.

!!! warning "Derivative target"

    drifft approximates $\mathrm D(A^kz)$, not the derivative of the executed
    approximation $V_m(V_m^TAV_m)^kV_m^Tz$ with its input-dependent basis.
    At finite dimension, these derivatives need not agree. The custom rule's
    target is the mathematical power of the represented operator $A$.

### Projected forward sensitivity

Applying Krylov projection to the coupled block operator approximates the
state and tangent together. The projection can depend on the supplied
perturbation, so this numerical calculation needs a dedicated AD rule.

Let $V_{2m,\mathrm{blk}}$ be the basis constructed for
$A_{\mathrm{blk}}$ with seed $z_{\mathrm{blk}}$, and define

$$
H_{2m,\mathrm{blk}}
=V_{2m,\mathrm{blk}}^T A_{\mathrm{blk}}
 V_{2m,\mathrm{blk}}.
$$

The projected action is

$$
y_{\mathrm{blk}}\approx
V_{2m,\mathrm{blk}}
H_{2m,\mathrm{blk}}^k
V_{2m,\mathrm{blk}}^T z_{\mathrm{blk}}.
$$

The basis has at most $2m$ columns, where $m$ is the configured `dimension`.
The `blk` label distinguishes this coupled projection from the primal
$V_m,H_m$ pair. Polynomial and rational methods use their respective
basis constructions from the [Krylov chapter](jacobi_krylov.md).
For dense rational block solves, write $S=I+\tau(I-A)$, $\tau=k/8$. Then

$$
I+\tau(I-A_{\mathrm{blk}})=
\begin{pmatrix}S&-\tau\dot A\\0&S\end{pmatrix}.
$$

Assuming $S$ is invertible, the inverse has the block form

$$
[I+\tau(I-A_{\mathrm{blk}})]^{-1}
=\begin{pmatrix}S^{-1}&\tau S^{-1}\dot A S^{-1}\\0&S^{-1}\end{pmatrix}.
$$

The implementation applies these inverses by solves, processing the lower
block first and reusing one LU factorization of $S$.

### Two-sided low-rank adjoint

The sum defining $\bar A$ contains a backward orbit under $A^T$ and a forward orbit
under $A$. Approximating each orbit in its own space gives a low-rank matrix
cotangent without storing a population-sized square matrix.

Choose orthonormal bases $V_{m,\mathrm L},V_{m,\mathrm R}$ seeded by $\bar y,z$, respectively, and define

$$
H_{m,\mathrm L}=V_{m,\mathrm L}^TA^TV_{m,\mathrm L},\quad H_{m,\mathrm R}=V_{m,\mathrm R}^TAV_{m,\mathrm R},\qquad
\bar y_{m,\mathrm L}=V_{m,\mathrm L}^T\bar y,\quad z_{m,\mathrm R}=V_{m,\mathrm R}^Tz.
$$

Each basis has at most $m$ columns; their actual dimensions may differ.
The labels $\mathrm L,\mathrm R$ identify the backward and forward spaces;
$\bar y_{m,\mathrm L},z_{m,\mathrm R}$ are their projected seeds.
Substitute the orbit approximations
$(A^T)^i\bar y\approx V_{m,\mathrm L}H_{m,\mathrm L}^i\bar y_{m,\mathrm L}$ and
$A^jz\approx V_{m,\mathrm R}H_{m,\mathrm R}^jz_{m,\mathrm R}$ into the cotangent sum:

$$
\bar A\approx V_{m,\mathrm L}CV_{m,\mathrm R}^T,\qquad
C=\sum_{j=0}^{k-1}
H_{m,\mathrm L}^{k-1-j}\bar y_{m,\mathrm L}z_{m,\mathrm R}^T(H_{m,\mathrm R}^T)^j.
$$

The same block-power identity shows that $C$ is the upper-right block of

$$
\begin{pmatrix}
H_{m,\mathrm L}&\bar y_{m,\mathrm L}z_{m,\mathrm R}^T\\0&H_{m,\mathrm R}^T
\end{pmatrix}^{k}.
$$

The transpose on the right block follows from transposing the forward orbit
in $\bar A$; omitting it would give a different matrix cotangent.
With $F=V_{m,\mathrm L}C$,

$$
\langle V_{m,\mathrm L}CV_{m,\mathrm R}^T,\dot A\rangle_F
=\sum_i F_i^T\dot A (V_{m,\mathrm R})_i,
\qquad
\bar\theta_A\approx\sum_i
\left(\frac{\partial[A(\theta)(V_{m,\mathrm R})_i]}{\partial\theta}\right)^TF_i.
$$

The operator pullback holds $F_i,(V_{m,\mathrm R})_i$ fixed during these contractions.
Differentiating their construction would introduce derivatives of the
approximation algorithm instead of evaluating this adjoint. The result is
returned in the operator's parameter structure without materializing $\bar A$.

## Encoding, reconstruction, and sources

A reduced-coordinate power is only part of the population calculation.
Parameter-dependent transforms and affine source terms contribute additional
chain-rule terms, even when the power derivative itself is evaluated exactly.

Write the prepared calculation as

$$
y=A^kz,\qquad v_{\mathrm{out}}=Ry+c.
$$

Its first variation is

$$
\boxed{
\dot v_{\mathrm{out}}
=\dot R\,y+R[\mathcal L_k(A,\dot A)z+A^k\dot z]+\dot c.}
$$

For incoming population cotangent $\bar v_{\mathrm{out}}$, the corresponding reverse
contributions are $\bar y=R^T\bar v_{\mathrm{out}}$, $\bar R=\bar v_{\mathrm{out}}y^T$, and
$\bar c=\bar v_{\mathrm{out}}$, followed by the power pullback. Ordinary JAX differentiation
propagates these contributions through the maps defining $A,z,R,c$.

### Jacobi coordinates

Jacobi's one-generation factorization determines both the reduced exponent
and the transforms surrounding it. Differentiating only the reduced matrix
would omit changes in how population mass enters and leaves coefficient space.

For $\widetilde P=DE$ and $H_{\mathrm J}=ED$,

$$
(\widetilde P^T)^gv_0
=E^T(H_{\mathrm J}^T)^{g-1}D^Tv_0,\qquad g\ge1.
$$

Thus $z=D^Tv_0$, $A=H_{\mathrm J}^T$, $k=g-1$, $R=E^T$, and $c=0$
without immigration. Their variations include

$$
\dot z=\dot D^Tv_0+D^T\dot v_0,\quad
\dot A=(\dot E D+E\dot D)^T,\quad \dot R=\dot E^T.
$$

Any parameter-independent transform has zero variation. Zero generations is
handled separately as an identity; population-coordinate evolution instead
uses $z=v_0$, $A=\widetilde P^T$, $k=g$, and $R=I$.

### Affine sources

A constant source can be included in a linear power by adding a constant
coordinate. This preserves the post-transition timing of immigration and
allows source sensitivities to use the same power rule.

$$
\begin{pmatrix}v_{t+1}\\1\end{pmatrix}
=\begin{pmatrix}\widetilde P^T&b\\0&1\end{pmatrix}
 \begin{pmatrix}v_t\\1\end{pmatrix}.
$$

The augmented operator's tangent is
$\left(\begin{smallmatrix}\dot{\widetilde P}^{\,T}&\dot b\\0&0\end{smallmatrix}\right)$;
the added constant coordinate has zero tangent.

In Jacobi coordinates, the encoded source is $D^Tb$. The population output
for $g\ge1$ is

$$
v_g=E^T\left[(H_{\mathrm J}^T)^{g-1}D^Tv_0
 +\sum_{j=0}^{g-2}(H_{\mathrm J}^T)^jD^Tb\right]+b,
$$

with an empty sum for $g=1$. Thus the source contributes through both the
augmented reduced operator and the final offset $c=b$. For changing sizes
or parameters, the first-variation and adjoint recurrences above compose
the interval and boundary operations directly.

## Implicit derivatives of linear solves

Expected visits are defined by a linear system, rather than a finite-time
power. Differentiating that defining equation avoids differentiating a
factorization or an iterative solver trajectory.

Write $\widetilde Q$ for the represented interior transition, retaining $Q$
for the exact restriction of $P$. Expected visits satisfy

$$
(I-\widetilde Q^T)u=e_1.
$$

Since the initial unit vector $e_1$ is fixed, the product rule gives

$$
-\dot{\widetilde Q}^{\,T}u+(I-\widetilde Q^T)\dot u=0,
\qquad
\boxed{(I-\widetilde Q^T)\dot u=\dot{\widetilde Q}^{\,T}u.}
$$

Given the incoming cotangent $\bar u$, define the adjoint solve variable
$\eta_u$ by $(I-\widetilde Q)\eta_u=\bar u$. Pairing with $\dot u$ yields

$$
\bar u^T\dot u
=\eta_u^T(I-\widetilde Q^T)\dot u
=\eta_u^T\dot{\widetilde Q}^{\,T}u
=\langle u\eta_u^T,\dot{\widetilde Q}\rangle_F.
$$

Therefore $\overline{\widetilde Q}=u\eta_u^T$. The variable $\eta_u$ is the
solution of an adjoint linear system, distinct from the incoming cotangent
$\bar u$; $u$ remains the expected-visits vector throughout.
The occupation interpretation requires transience, and the represented system
must be invertible. The derivative inherits solve and representation errors.

Lineax supplies these implicit rules. Shifted solves also occur in rational
Krylov construction, but the outer power derivative uses the custom
sensitivity rules above, not differentiation of the Arnoldi basis.

## Implementation correspondence

The mathematical power rules enter JAX through custom primitives, while
ordinary differentiation connects them to biological parameters and coordinate
transforms. This boundary keeps adaptive numerical construction separate from
the derivative target.

All drifft-owned custom differentiation uses
`jax.experimental.hijax.HiPrim`. HiJAX derives linearization and reverse mode
from JVP rules, with explicit transposes for linear inputs. The Krylov Fréchet
primitive supplies its operator pullback explicitly and rejects unsupported
higher derivatives. Dense's binomial operation uses ordinary JAX AD.

The primal power is computed with convergence diagnostics and passed through
a cached-power primitive. Its derivative is attached to $A$ and $z$; the
cached value and primal Arnoldi basis are not differentiated. Preparation and
reconstruction remain connected to parameters through ordinary JAX operations.

## Accuracy and limitations

Exact derivative identities do not make their projected evaluations exact.
The supported derivative orders and convergence checks depend on the evolution
method, and must be distinguished from backend approximation error.

| Calculation | Derivative target | Higher-order support |
| --- | --- | --- |
| Sequential | Repeated numerical transition, including its backend approximation | Supported through the recurrence and supported backend rules |
| PolynomialKrylov / RationalKrylov | Approximate sensitivity of the prepared power $A^kz$ | Higher-order differentiation through the Fréchet rule raises explicitly |

For the coupled forward action, the combined error estimate must satisfy both
state and tangent tolerances. Each block uses
$\mathtt{atol}+\mathtt{rtol}\,\|\cdot\|_1$; the smaller tolerance is applied
to the combined estimate so a large state cannot mask a small sensitivity.
The reverse operator rule compares nested projected cotangents in Frobenius
norm and checks intermediate orbit recurrence defects. These checks can detect
unresolved propagation even when nested approximations agree.

!!! note "Diagnostic estimates, not error bounds"

    Recurrence-defect estimates use unit amplification rather than bounds on
    subsequent operator powers. For general nonnormal operators, they are
    heuristic. Neither these estimates nor primal convergence certifies
    derivative accuracy for the exact Wright–Fisher process.

Zero seeds and invariant-subspace breakdown are supported when sensitivity
checks pass. A failed sensitivity or pullback raises even with `throw=False`;
a failing member under `vmap` can fail the batch. There is no automatic fallback.
Numerical error estimates and status fields are not differentiation targets.

The [accuracy guide](../guide/numerics.md) covers validation. In particular,
finite differences of an underresolved Krylov primal can disagree with its
custom derivative because they differentiate different finite approximations.

## References

The implementation follows the matrix-function sensitivity framework above;
the references provide the low-rank approximation background and the AD
interfaces used to express it.

- [Kandolf et al., low-rank Fréchet derivative approximations](https://arxiv.org/abs/2008.12926).
- [JAX custom derivative rules and HiJAX](https://docs.jax.dev/en/latest/301/custom-derivatives.html).
- [Lineax linear solves](https://docs.kidger.site/lineax/api/linear_solve/).
