# Krylov power methods

An evolution method controls how a prepared transition is applied over many
generations. Polynomial and rational Krylov methods approximate the power
acting on the supplied state, using a smaller projected problem.

## Problem and representation

Let $A$ be the prepared forward operator, $p$ its working-state vector, and
$g$ the required exponent. The task is to approximate $A^g p$. The working-state
dimension $d$ is population-sized for FFT/NUFFT and reduced for Jacobi.
A basis with at most $k$ columns defines a projected matrix $B=V^TAV$ and
an approximation $VB^gV^Tp$. `dimension` sets the maximum basis size.

For Jacobi's one-step factorization $P\approx DE$ and $H=ED$, the corresponding
ideal forward action through $g\ge1$ generations is

$$
p_g\approx E^T(H^T)^{g-1}D^Tp_0.
$$

Thus the reduced power exponent is $g-1$. The
[Jacobi chapter](jacobi.md) describes the transforms and their approximation
errors separately from the power algorithm.

## Computation

Polynomial Krylov constructs a basis using repeated applications of $A$.
Rational Krylov constructs a basis using shifted solves with

$$
[I+\tau(I-A)]^{-1},\qquad \tau=g/8,
$$

where $g$ is the prepared power exponent. The implementation reorthogonalizes
twice, projects the original operator $A$, and evaluates the projected power
by binary powering. It does not power the shifted inverse as the output model.

Jacobi uses a reusable dense LU factorization for these solves. FFT and NUFFT
use matrix-free, unpreconditioned GMRES. Both paths approximate the same kind
of power action, but their setup and inner-iteration costs differ.

For constant immigration $b$ with Jacobi, initialize $y_0=D^Tp_0$ and use

$$
y_{t+1}=H^Ty_t+D^Tb,\qquad p_g\approx E^Ty_{g-1}+b.
$$

The implementation represents this affine recurrence by an augmented matrix.
The source adds a coordinate to the working state; the power-method logic
otherwise applies to that augmented operator.

## Cost and reuse

Let $T_A$ be the cost of an operator application and $T_S$ the cost of one shifted
solve after preparation. The main costs, excluding external encoding/decoding,
are:

| Stage | Polynomial Krylov | Rational Krylov |
| --- | --- | --- |
| Basis storage | $O(dk)$ | $O(dk)$ |
| Basis construction and projection | $O(kT_A+dk^2)$ | $O(kT_S+kT_A+dk^2)$ |
| Projected binary power | $O(k^3\log g)$ for $g\ge2$ | $O(k^3\log g)$ for $g\ge2$ |

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

JAX differentiates operator actions, frequency-map parameters, and the
input-dependent basis. FFT/NUFFT shifted solves use implicit differentiation
through Lineax; a failed tangent or transpose solve raises even when the primal call used
`throw=False`. The accumulated recurrence-defect diagnostic is excluded from
differentiation.

Derivatives at zero seeds or changes in Arnoldi breakdown are not guaranteed.
Differentiation can retain substantially more memory than a forward action.
Because the basis depends on the input, the methods do not expose a matched
expectation transpose. Use Sequential for `model.expect()`.

## Use in drifft

Select `PolynomialKrylov` or `RationalKrylov` as the method of a
`DemographicTransition`, or apply it directly to a prepared backend. Both
support FFT, NUFFT, and Jacobi; frequency-map support follows the backend.
Population-size boundary steps still use FFT/NUFFT.

See [choosing algorithms](../guide/backends.md) for recommendations,
[direct epoch operations](epochs.md) for timing and immigration semantics, and
the [method API](../api/methods.md) for constructors and result types.
