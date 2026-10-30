# Krylov power methods

An evolution method controls how a prepared transition is applied over many
generations. Polynomial and rational Krylov methods approximate the power
acting on the supplied state, using a smaller projected problem.

For configuration, see the [API reference](../api/methods.md); for a workflow, see
[choosing algorithms](../guide/backends.md). Notation follows the
[shared conventions](../guide/notation.md). The
[computational overview](overview.md) places this method within the full calculation.

## Problem and representation

Krylov methods approximate a long evolution in a basis adapted to the initial
state. The projected power is small even when the working state is population-sized.

Let $A$ be the prepared forward operator, $z$ its working-state vector, and
$k$ the required exponent. The task is to approximate $A^k z$. The working-state
dimension $d$ is population-sized for FFT, NUFFT, and Dense, and reduced for Jacobi.
A basis with at most $m$ columns defines a projected matrix $H_m=V_m^TAV_m$ and
an approximation $V_mH_m^kV_m^Tz$. `dimension` sets the maximum basis size.

For Jacobi's one-step factorization $\widetilde P=DE$ and $H_{\mathrm J}=ED$, the corresponding
ideal forward action through $g\ge1$ generations is

$$
v_g\approx E^T(H_{\mathrm J}^T)^{g-1}D^Tv_0.
$$

Thus the reduced power exponent is $k=g-1$. The
[Jacobi chapter](jacobi.md) describes the transforms and their approximation
errors separately from the power algorithm.

## Computation

Polynomial and rational Krylov differ in how they build the approximation
space. Both project the original forward operator and power that projection.

Polynomial Krylov constructs a basis using repeated applications of $A$.
Rational Krylov constructs a basis using shifted solves with

$$
[I+\tau(I-A)]^{-1},\qquad \tau=k/8,
$$

where $k$ is the prepared power exponent. The implementation reorthogonalizes
twice, projects the original operator $A$, and evaluates the projected power
by binary powering. It does not power the shifted inverse as the output model.

Jacobi uses a reusable dense LU factorization for these solves. Other transition
backends use unpreconditioned GMRES through their operator interface. Both paths
approximate the same kind of power action, but their setup and inner-iteration
costs differ.

For constant immigration $b$ with Jacobi, initialize $z_0=D^Tv_0$ and use

$$
z_{t+1}=H_{\mathrm J}^Tz_t+D^Tb,\qquad v_g\approx E^Tz_{g-1}+b.
$$

The implementation represents this affine recurrence by an augmented matrix.
The source adds a coordinate to the working state; the power-method logic
otherwise applies to that augmented operator.

## Cost and reuse

The benefit depends on how many basis vectors are needed compared with the
number of generations. Rational Krylov also pays for shifted solves.

Let $T_A$ be the cost of an operator application and $T_S$ the cost of one shifted
solve after preparation. The main costs, excluding external encoding/decoding,
are:

| Stage | Polynomial Krylov | Rational Krylov |
| --- | --- | --- |
| Basis storage | $O(dm)$ | $O(dm)$ |
| Basis construction and projection | $O(mT_A+dm^2)$ | $O(mT_S+mT_A+dm^2)$ |
| Projected binary power | $O(m^3\log k)$ for $k\ge2$ | $O(m^3\log k)$ for $k\ge2$ |

For dense Jacobi matrices, factorization adds $O(d^3)$ and each shifted solve
costs $O(d^2)$. GMRES solve cost instead depends on its inner iterations and
backend applications. These costs are additional to the backend geometry and
transforms described in the backend chapters.

The saving comes from replacing many generation-by-generation applications by
a smaller basis calculation. Basis size can grow with interval difficulty;
there is no uniform speedup independent of model, duration, or tolerance.
The basis depends on the input state, and shifted factors depend on the operator
and exponent. Do not assume either is reusable after those inputs change.

## Accuracy and controls

Basis size and tolerances control the power approximation. A converged power
action still inherits any error in the underlying transition backend.

| Control | Role |
| --- | --- |
| `dimension` | Maximum retained Krylov basis size |
| `rtol`, `atol` | Acceptance threshold for the power-action estimate |
| `throw` on application | Raise on numerical failure or return status for inspection |

The estimate combines a smaller-basis comparison with a recurrence-defect check.
Acceptance requires the estimate to satisfy
`atol + rtol * sum(abs(distribution))`. It is a diagnostic of the power action,
not a bound for all errors in the biological calculation. Increase backend
accuracy and basis dimension independently.

Neither method automatically restarts or falls back to Sequential. An
unsuccessful inner GMRES solve is reported as a linear-solve failure; increasing
the outer basis dimension does not repair that solve. Check `converged` and
`status` when using `throw=False`. The [accuracy guide](../guide/numerics.md)
describes validation of the final quantity.

## Differentiation and adjoints

Krylov differentiation targets the mathematical power sensitivity; it does
not differentiate the input-dependent basis construction. This distinction
matters when validating gradients at finite basis dimension.

!!! note "Advanced: power sensitivities"

    The [autodiff chapter](autodiff.md) derives the coupled forward action and
    low-rank reverse action, and specifies their convergence and higher-order
    limitations.

The primal approximation is input dependent and has no matched expectation
transpose. Use Sequential when you need `model.expect()`.
