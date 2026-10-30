"""Krylov approximations of mathematical power sensitivities and pullbacks."""

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from .problem import EpochResult, _epoch_result, _working_problem
from .projection import _matrix_power_matvec, _projected_result, project


def _sensitivity_projection(
    method, operator, initial, steps, *, dimension=None, shifted_solve=None
):
    """Reuse the method's Arnoldi space, retaining its basis for adjoint factors."""
    problem = _working_problem(operator, initial, steps)
    dimension = method.dimension if dimension is None else dimension
    return project(
        problem,
        dimension,
        polynomial=method._polynomial,
        return_basis=True,
        shifted_solve=shifted_solve,
    )


def _frechet_power(method, operator, perturbation, initial, tangent_initial, steps):
    """Approximate the power sensitivity by a coupled block operator action.

    This numerical kernel is not a custom JVP: its adaptive space can depend on
    the supplied direction. AD integration requires a dedicated tangent primitive.
    """
    if steps == 0:
        return _epoch_result(tangent_initial, jnp.asarray(0.0), method.atol, False)
    size = initial.size
    shifted_solve = None
    if isinstance(operator, lx.MatrixLinearOperator):
        a, e = operator.matrix, perturbation.as_matrix()
        # [[A, E], [0, A]]**g @ [dx; x] has derivative d(A**g x) in its
        # upper block. This computes the mathematical directional derivative
        # without differentiating how Arnoldi chose its basis.
        block = lx.MatrixLinearOperator(jnp.block([[a, e], [jnp.zeros_like(a), a]]))
        if not method._polynomial:
            shift = steps / 8
            shifted = lx.MatrixLinearOperator(
                jnp.eye(size) + shift * (jnp.eye(size) - a)
            )
            solver = lx.LU()
            factor = solver.init(shifted, {})

            def shifted_solve(rhs):
                # Both diagonal blocks share the same shifted A. Reuse one LU,
                # solve the lower block first, then feed its coupling upward.
                lower = lx.linear_solve(
                    shifted, rhs[size:], solver, state=factor, throw=False
                )
                upper = lx.linear_solve(
                    shifted,
                    rhs[:size] + shift * perturbation.mv(lower.value),
                    solver,
                    state=factor,
                    throw=False,
                )
                success = (lower.result == lx.RESULTS.successful) & (
                    upper.result == lx.RESULTS.successful
                )
                return jnp.concatenate((upper.value, lower.value)), success
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
        method,
        block,
        seed,
        steps,
        dimension=2 * method.dimension,
        shifted_solve=shifted_solve,
    )
    fine = projection.fine
    # Require the combined error to satisfy each output block's tolerance. A large
    # primal must not mask an inaccurate, small sensitivity. This remains an estimate.
    tolerance = jnp.minimum(
        method.atol + method.rtol * jnp.sum(jnp.abs(fine[:size])),
        method.atol + method.rtol * jnp.sum(jnp.abs(fine[size:])),
    )
    result = _projected_result(
        _working_problem(block, seed, steps),
        projection,
        0.0,
        tolerance,
        False,
    )
    return EpochResult(
        result.distribution[:size],
        result.error_estimate,
        result.converged,
        result.status,
    )


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


def _frechet_adjoint_factors(method, operator, initial, cotangent, steps):
    """Approximate the operator pullback as left @ right.T without forming it.

    The two spaces use A.T and A, seeded by the output cotangent and input state.
    Nested-space differences and intermediate recurrence defects estimate
    Frobenius error; they are not certificates.
    """
    if steps == 0:
        empty = jnp.zeros((initial.size, 0), dtype=initial.dtype)
        return empty, empty, jnp.asarray(0.0), jnp.asarray(True)
    left_projection = _sensitivity_projection(method, operator.T, cotangent, steps)
    right_projection = _sensitivity_projection(method, operator, initial, steps)
    u, v = left_projection.basis, right_projection.basis
    assert u is not None and v is not None
    h, j = left_projection.projected, right_projection.projected
    c, x = left_projection.coordinates, right_projection.coordinates
    core = _frechet_core(h, j, c, x, steps)
    full = u.shape[1] == initial.size and v.shape[1] == initial.size
    exact_polynomial = method._polynomial and min(u.shape[1], v.shape[1]) >= steps
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
        error = jnp.linalg.norm(difference)
    # A nested difference can vanish before either projected orbit reaches the
    # part of the spectrum contributing to the pullback. Track intermediate
    # recurrence defects as well. Unit amplification mirrors the primal defect
    # diagnostic; for general nonnormal operators this remains a heuristic,
    # rather than a certified error bound.
    left_weights = jnp.linalg.norm(left_projection.defect, axis=0)
    right_weights = jnp.linalg.norm(right_projection.defect, axis=0)

    def orbit_step(carry, _):
        left_state, right_state, left_error, right_error = carry
        diagnostics = (
            jnp.linalg.norm(left_state),
            jnp.linalg.norm(right_state),
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
        jnp.linalg.norm(cotangent - u @ c),
        jnp.linalg.norm(initial - v @ x),
    )
    _, (left_norms, right_norms, left_errors, right_errors) = jax.lax.scan(
        orbit_step, (c, x, *seed_errors), None, length=steps
    )
    recurrence_error = jnp.sum(
        left_errors * (right_norms[::-1] + right_errors[::-1])
        + left_norms * right_errors[::-1]
    )
    error = jnp.maximum(error, recurrence_error)
    left = u @ core
    # Zero columns at invariant-subspace breakdown need no completion: the seed's
    # orbit is represented by the remaining basis columns on each side.
    finite = (
        jnp.all(jnp.isfinite(left)) & jnp.all(jnp.isfinite(v)) & jnp.isfinite(error)
    )
    converged = (
        left_projection.success
        & right_projection.success
        & finite
        & (error <= method.atol + method.rtol * jnp.linalg.norm(core))
    )
    return left, v, error, converged


def _frechet_operator_pullback(
    method, operator, tangent_template, initial, cotangent, steps
):
    """Contract low-rank adjoint factors into the active operator PyTree leaves.

    The tangent template selects perturbed leaves, not biological parameter names.
    The numerical factors are residuals of the inner operator VJP; differentiating
    their construction there would introduce unwanted basis-derivative terms.
    """

    def is_none(x):
        return x is None

    if not any(
        x is not None for x in jax.tree.leaves(tangent_template, is_leaf=is_none)
    ):
        return (
            jax.tree.map(lambda _: None, operator),
            jnp.asarray(0.0),
            jnp.asarray(True),
        )
    selected = jax.tree.map(lambda x: x is not None, tangent_template, is_leaf=is_none)
    active, fixed = eqx.partition(operator, selected, is_leaf=is_none)
    left, right, error, converged = _frechet_adjoint_factors(
        method, operator, initial, cotangent, steps
    )

    def action(active):
        op = eqx.combine(active, fixed)
        return jax.lax.map(op.mv, right.T)

    _, pullback = eqx.filter_vjp(action, active)
    return pullback(left.T)[0], error, converged
