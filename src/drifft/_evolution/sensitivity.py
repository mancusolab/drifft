r"""Numerically approximate power derivatives and operator-adjoint factors.

For $y=A^k x$, a perturbation $E$ of $A$ and an initial-state direction $\dot x$
can be propagated together using the block identity

$$
\begin{pmatrix}A&E\\0&A\end{pmatrix}^{k}
\begin{pmatrix}\dot x\\x\end{pmatrix}
=
\begin{pmatrix}
A^k\dot x+\sum_{j=0}^{k-1}A^{k-1-j}EA^j x\\ A^k x
\end{pmatrix}.
$$

This module evaluates the coupled recurrence directly or approximates the
upper block using Krylov projection. For an output cotangent $g$, the operator
adjoint is
$G=\sum_{j=0}^{k-1}(A^T)^{k-1-j}g\,(A^j x)^T$.
Separate projected spaces seeded by $g$ and $x$ approximate it as $LR^T$,
avoiding construction of the full matrix cotangent. Operator-specific code can
then contract these fixed factors into parameter coordinates.

These kernels accept explicit projection settings and tolerances, without
requiring an evolution-method object. They do not define JAX derivative rules.
Projected bases can depend on the state, cotangent or perturbation direction.
Calling these kernels inside the primitives in `autodiff.py` approximates the
mathematical power derivative without differentiating the basis-selection algorithm.
`projection.py` supplies the shared projection machinery; `operators.py` and
`parameterized.py` supply parameter contractions after factor evaluation.

Nested-space comparisons and recurrence defects estimate sensitivity error;
they are not certified bounds and exclude backend representation error. The
method capabilities enforce convergence separately from the primal solve. Further
Krylov operator sensitivities are not obtained by applying AD to these kernels.
"""

import jax
import jax.numpy as jnp
import lineax as lx

from .problem import _epoch_result, _relative_l1_budget, _working_problem
from .projection import (
    _dense_shifted_solve,
    _matrix_power_matvec,
    _norm_scale,
    _rational_shift,
    project,
)


def _sequential_power_pair(operator, perturbation, initial, tangent, steps):
    """Evaluate A**g x and its joint derivative in one coupled recurrence."""

    def step(_, carry):
        state, direction = carry
        return operator.mv(state), jax.tree.map(
            jnp.add, operator.mv(direction), perturbation.mv(state)
        )

    return jax.lax.fori_loop(0, steps, step, (initial, tangent))


def _sensitivity_projection(
    operator,
    initial,
    steps,
    *,
    dimension,
    polynomial,
    shifted_solve=None,
):
    """Build the requested Arnoldi space, retaining its basis for adjoint factors."""
    problem = _working_problem(operator, steps)
    return project(
        problem,
        initial,
        dimension,
        polynomial=polynomial,
        return_basis=True,
        shifted_solve=shifted_solve,
    )


def _frechet_power(
    operator,
    perturbation,
    initial,
    tangent_initial,
    steps,
    *,
    dimension,
    rtol,
    atol,
    polynomial,
    check_error=True,
    direction_scale=1.0,
):
    """Approximate the power sensitivity by a coupled block operator action.

    This numerical kernel is not a custom JVP: its adaptive space can depend on
    the supplied direction. AD integration requires a dedicated tangent primitive.

    `dimension` is the single-state projection budget; the coupled block uses
    twice that budget. `polynomial` selects polynomial rather than rational
    projection. Both output blocks must satisfy their own `atol + rtol * L1`
    tolerance, so a large primal cannot mask an inaccurate tangent.
    `direction_scale` restores a jointly normalized perturbation and state direction
    to their original units before applying the tangent tolerance.
    With `check_error=False`, skip error estimation and retain only inner-solve
    and finite-output checks; the returned error estimate is zero (not computed).
    """
    if steps == 0:
        return _epoch_result(
            direction_scale * tangent_initial, jnp.asarray(0.0), atol, False
        )
    size = initial.size
    shifted_solve = None
    if isinstance(operator, lx.MatrixLinearOperator):
        a, e = operator.matrix, perturbation.as_matrix()
        # [[A, E], [0, A]]**g @ [dx; x] has derivative d(A**g x) in its
        # upper block. This computes the mathematical directional derivative
        # without differentiating how Arnoldi chose its basis.
        block = lx.MatrixLinearOperator(jnp.block([[a, e], [jnp.zeros_like(a), a]]))
        if not polynomial:
            shift = _rational_shift(steps)
            solve = _dense_shifted_solve(a, steps)

            def shifted_solve(rhs):
                # Both diagonal blocks share the same shifted A. Solve the
                # lower block first, then reuse its factorization for the upper.
                lower = solve(rhs[size:])
                upper = solve(rhs[:size] + shift * perturbation.mv(lower.value))
                return lower._replace(
                    value=jnp.concatenate((upper.value, lower.value)),
                    success=lower.success & upper.success,
                )
    else:

        def action(state):
            tangent, primal = state[:size], state[size:]
            return jnp.concatenate(
                (operator.mv(tangent) + perturbation.mv(primal), operator.mv(primal))
            )

        block = lx.FunctionLinearOperator(
            action, jax.ShapeDtypeStruct((2 * size,), initial.dtype)
        )
    seed = jnp.concatenate((tangent_initial, initial))
    projection = _sensitivity_projection(
        block,
        seed,
        steps,
        dimension=2 * dimension,
        polynomial=polynomial,
        shifted_solve=shifted_solve,
    )
    fine = jnp.concatenate(
        (direction_scale * projection.fine[:size], projection.fine[size:])
    )
    if not check_error:
        # Check both blocks even though only the tangent is returned.
        result = _epoch_result(fine, jnp.asarray(0.0), 0.0, False, projection.success)
        return result._replace(distribution=result.distribution[:size])
    # Keep separate L1 budgets for [dx, x]. A lower-block defect also
    # drives the upper block via E on each remaining step. Estimate this
    # transfer on the defect columns, assuming unit propagation by A as in
    # the primal estimator. This avoids materializing a matrix-free E.
    tangent_defect, primal_defect = jnp.split(projection.defect, 2)
    weights = jnp.stack(
        (
            jnp.sum(jnp.abs(tangent_defect), axis=0),
            jnp.sum(jnp.abs(primal_defect), axis=0),
        )
    )
    transfer = jax.lax.map(
        lambda d: jnp.sum(jnp.abs(perturbation.mv(d))), primal_defect.T
    )

    def accumulate(index, carry):
        state, errors = carry
        coefficients = jnp.abs(state)
        errors = errors + weights @ coefficients
        errors = errors.at[0].add((steps - 1 - index) * (transfer @ coefficients))
        return projection.projected @ state, errors

    _, errors = jax.lax.fori_loop(
        0, steps, accumulate, (projection.coordinates, jnp.zeros(2, dtype=fine.dtype))
    )
    if projection.coarse is not None:
        difference = (projection.fine - projection.coarse).reshape(2, size)
        errors = jnp.maximum(errors, jnp.sum(jnp.abs(difference), axis=1))
    # Restore the caller's units before checking the tangent's absolute budget.
    errors = errors.at[0].multiply(direction_scale)
    tolerances = atol + jax.vmap(_relative_l1_budget, in_axes=(0, None))(
        fine.reshape(2, size), rtol
    )
    result = _epoch_result(fine, errors, tolerances, False, projection.success)
    return result._replace(distribution=fine[:size])


def _frechet_core(left_matrix, right_matrix, left_seed, right_seed, steps):
    """Compute the small projected Frechet adjoint, with the right block transposed."""
    n, m = left_seed.size, right_seed.size
    # The upper-right block of this power sums H**k c x.T (J.T)**(g-1-k).
    # Lifting it with the left/right bases yields the matrix-gradient factors;
    # omitting J.T would reverse the orientation of the right orbit.
    block = jnp.block(
        [
            [left_matrix, jnp.outer(left_seed, right_seed)],
            [jnp.zeros((m, n)), right_matrix.T],
        ]
    )
    seed = jnp.concatenate((jnp.zeros((n, m)), jnp.eye(m)), axis=0)
    return _matrix_power_matvec(block, seed, steps)[:n]


def _scaled_norm(value):
    """Compute a Euclidean or Frobenius norm without squaring unscaled entries."""
    scale = _norm_scale(value)
    return scale * jnp.linalg.norm(value / scale)


def _frechet_adjoint_factors(
    operator,
    initial,
    cotangent,
    steps,
    *,
    dimension,
    rtol,
    atol,
    polynomial,
    check_error=True,
):
    """Approximate the operator pullback as left @ right.T without forming it.

    The two spaces use A.T and A, seeded by the output cotangent and input state.
    Nested-space differences and intermediate recurrence defects estimate
    Frobenius error; they are not certificates.

    Each space uses the supplied `dimension` and `polynomial` mode. Acceptance
    compares the error estimate with `atol + rtol * norm(core)`, where `core`
    is the projected adjoint and the norm is Frobenius. Return the two factors,
    error estimate, and convergence flag; callers decide how to handle failure.
    With `check_error=False`, skip error estimation and retain only inner-solve
    and finite-factor checks; the returned error estimate is zero (not computed).
    """
    if steps == 0:
        empty = jnp.zeros((initial.size, 0), dtype=initial.dtype)
        return empty, empty, jnp.asarray(0.0), jnp.asarray(True)
    left_projection = _sensitivity_projection(
        operator.T, cotangent, steps, dimension=dimension, polynomial=polynomial
    )
    right_projection = _sensitivity_projection(
        operator, initial, steps, dimension=dimension, polynomial=polynomial
    )
    u, v = left_projection.basis, right_projection.basis
    assert u is not None and v is not None
    h, j = left_projection.projected, right_projection.projected
    c, x = left_projection.coordinates, right_projection.coordinates
    core = _frechet_core(h, j, c, x, steps)
    left = u @ core
    finite = jnp.all(jnp.isfinite(left)) & jnp.all(jnp.isfinite(v))
    numerical_success = left_projection.success & right_projection.success & finite
    if not check_error:
        return left, v, jnp.asarray(0.0), numerical_success
    full = u.shape[1] == initial.size and v.shape[1] == initial.size
    exact_polynomial = polynomial and min(u.shape[1], v.shape[1]) >= steps
    if full or exact_polynomial:
        error = jnp.asarray(0.0)
    else:
        left_half, right_half = max(1, u.shape[1] // 2), max(1, v.shape[1] // 2)
        coarse = _frechet_core(
            h[:left_half, :left_half],
            j[:right_half, :right_half],
            c[:left_half],
            x[:right_half],
            steps,
        )
        difference = core.at[:left_half, :right_half].add(-coarse)
        error = _scaled_norm(difference)
    # A nested difference can vanish before either projected orbit reaches the
    # part of the spectrum contributing to the pullback. Track intermediate
    # recurrence defects as well. Unit amplification mirrors the primal defect
    # diagnostic; for general nonnormal operators this remains a heuristic,
    # rather than a certified error bound.
    left_weights = jax.vmap(_scaled_norm, in_axes=1)(left_projection.defect)
    right_weights = jax.vmap(_scaled_norm, in_axes=1)(right_projection.defect)

    def orbit_step(carry, _):
        left_state, right_state, left_error, right_error = carry
        diagnostics = (
            _scaled_norm(left_state),
            _scaled_norm(right_state),
            left_error,
            right_error,
        )
        following = (
            h @ left_state,
            j @ right_state,
            left_error + left_weights @ jnp.abs(left_state),
            right_error + right_weights @ jnp.abs(right_state),
        )
        return following, diagnostics

    seed_errors = (
        _scaled_norm(cotangent - u @ c),
        _scaled_norm(initial - v @ x),
    )
    _, (left_norms, right_norms, left_errors, right_errors) = jax.lax.scan(
        orbit_step, (c, x, *seed_errors), None, length=steps
    )
    recurrence_error = jnp.sum(
        left_errors * (right_norms[::-1] + right_errors[::-1])
        + left_norms * right_errors[::-1]
    )
    error = jnp.maximum(error, recurrence_error)
    # Zero columns at invariant-subspace breakdown need no completion: the seed's
    # orbit is represented by the remaining basis columns on each side.
    converged = (
        numerical_success
        & jnp.isfinite(error)
        & (error <= atol + rtol * _scaled_norm(core))
    )
    return left, v, error, converged
