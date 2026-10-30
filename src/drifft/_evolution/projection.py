"""Arnoldi projection kernels and reconstructed power-error estimates."""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax.experimental import hijax

from .problem import _epoch_result, _PowerProblem


class ProjectionResult(NamedTuple):
    """Projected power and diagnostics in working coordinates.

    For basis V and operator images AV, `projected` is V.T @ AV,
    `coordinates` is V.T @ seed, and `defect` is AV - V @ projected.
    `fine` and `coarse` are lifted powers in full and nested spaces; `success`
    records inner-solve validity, not outer power convergence. `basis` is None
    unless explicitly requested for sensitivities, avoiding unnecessary retention.
    """

    fine: jax.Array
    coarse: jax.Array
    defect: jax.Array
    projected: jax.Array
    coordinates: jax.Array
    success: jax.Array
    basis: jax.Array | None


def _orthogonalize(basis, candidate):
    """Reorthogonalize twice and return zero at invariant-subspace breakdown."""
    original_squared = candidate @ candidate
    # Two passes recover orthogonality when the first subtraction loses relative
    # accuracy. Keep zero columns at breakdown instead of dividing roundoff by
    # a tiny norm; invariant subspaces need no arbitrary orthogonal completion.
    for _ in range(2):
        candidate = candidate - basis @ (basis.T @ candidate)
    squared = candidate @ candidate
    active = squared > (64 * jnp.finfo(candidate.dtype).eps) ** 2 * original_squared
    denominator = jnp.sqrt(jnp.where(active, squared, 1.0))
    return jnp.where(active, candidate / denominator, 0.0)


def _project_basis(basis, images, seed, steps, *, exact, success, return_basis):
    """Project A itself and compare nested spaces while retaining raw defects."""
    projected = basis.T @ images
    coordinates = basis.T @ seed
    fine = basis @ _matrix_power_matvec(projected, coordinates, steps)
    fine = _guard_krylov_derivative(fine, basis, seed @ seed)
    defect = images - basis @ projected
    if exact:
        coarse = fine
    else:
        half = max(1, basis.shape[1] // 2)
        coarse = basis[:, :half] @ _matrix_power_matvec(
            projected[:half, :half], coordinates[:half], steps
        )
    return ProjectionResult(
        fine,
        coarse,
        defect,
        projected,
        coordinates,
        success,
        basis if return_basis else None,
    )


def _matrix_power_matvec(matrix: jax.Array, vector: jax.Array, steps: int):
    """Apply a nonnegative static power without assembling the powered matrix.

    Square the reduced matrix, but accumulate set bits into a vector. This
    replaces the matrix-matrix products used to combine powers with matvecs.
    The Python loop stages only O(log steps) operations and supports reverse AD.
    """
    # `steps` is static: this short loop unrolls during tracing and supports
    # both forward and reverse AD. CPU benchmarks found rolled lax.scan slower
    # for gradients; fully unrolled scan matched warm speed but compiled slower.
    # Keep Python here; see benchmarks/POWER_LOOPS.md for the unrolling sweep.
    while steps:
        if steps & 1:
            vector = matrix @ vector
        steps >>= 1
        if steps:
            matrix = matrix @ matrix
    return vector


def _require_krylov_derivative_jvp(primals, tangents):
    """Keep the failure check in both the tangent and transposed tangent paths."""
    value, supported = primals
    tangent, _ = tangents
    tangent = hijax.instantiate_zeros(tangent)
    # A checked residual multiplier survives transposition to reverse mode.
    # Checking only the primal result can be eliminated when requesting grad.
    scale = eqx.error_if(
        jnp.ones((), dtype=value.dtype),
        ~supported,
        "Krylov derivatives require a nonzero seed and no Arnoldi breakdown; "
        "use Sequential for derivatives in this regime. "
        "throw=False applies only to forward evolution failures.",
    )
    return _require_krylov_derivative(value, supported), scale * tangent


class _RequireKrylovDerivative(hijax.HiPrim):
    """Preserve the Arnoldi derivative guard through JVP and reverse mode."""

    def __init__(self, value, supported):
        """Declare an identity operation with a dynamic support predicate."""
        self.in_avals = (value, supported)
        self.out_aval = value
        self.params = {}
        super().__init__()

    def expand(self, *args):
        """Return the unchecked forward value."""
        value, _ = args
        return value

    def jvp(self, primals, tangents):
        """Check support on the tangent path."""
        return _require_krylov_derivative_jvp(primals, tangents)

    lin, linearized = hijax.linearize_from_jvp
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

    def batch_dim_rule(self, axis_data, dims):
        """Map the result when either input is mapped."""
        return 0 if any(dim is not None for dim in dims) else None


def _require_krylov_derivative(value, supported):
    """Bind the derivative-only support check."""
    return _RequireKrylovDerivative(jax.typeof(value), jax.typeof(supported))(
        value, supported
    )


def _guard_krylov_derivative(value, basis, norm_squared):
    """Reject zero seeds and any zero column left by the breakdown branch."""
    supported = (norm_squared > 0) & jnp.all(jnp.sum(basis * basis, axis=0) > 0)
    return _require_krylov_derivative(value, supported)


class _ShiftedSolveResult(NamedTuple):
    """Checked resolvent application with solver-specific diagnostic counters."""

    value: jax.Array
    success: jax.Array
    num_steps: jax.Array
    residual_norm: jax.Array


def _rational_shift(steps: int) -> float:
    """Choose the common resolvent shift for a prepared exponent."""
    return steps / 8


def _dense_shifted_solve(matrix, steps):
    """Factor the dense resolvent once and return checked, reusable solves."""
    identity = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    shifted = lx.MatrixLinearOperator(
        identity + _rational_shift(steps) * (identity - matrix)
    )
    solver = lx.LU()
    factor = solver.init(shifted, {})

    def solve(rhs):
        """Reuse LU state and retain failure status without throwing inside Arnoldi."""
        result = lx.linear_solve(shifted, rhs, solver, state=factor, throw=False)
        return _ShiftedSolveResult(
            result.value,
            result.result == lx.RESULTS.successful,
            jnp.asarray(0),
            jnp.asarray(0.0, dtype=rhs.dtype),
        )

    return solve


def _rational_projection(problem, dimension, shifted_solve, *, return_basis):
    """Build one resolvent Arnoldi space using a checked solve callback."""
    seed = problem.initial
    dimension = min(seed.size, dimension)
    # Zero seeds and invariant-subspace breakdown leave zero columns. Preserve
    # that convention for both dense and matrix-free resolvent applications.
    norm_squared = seed @ seed
    norm = jnp.sqrt(jnp.where(norm_squared > 0, norm_squared, 1.0))
    basis = jnp.zeros((seed.size, dimension), dtype=seed.dtype)
    basis = basis.at[:, 0].set(seed / norm)

    def step(index, carry):
        """Apply the resolvent and accumulate checks before reorthogonalization."""
        basis, success, num_steps, max_residual = carry
        solution = shifted_solve(basis[:, index])
        success = success & solution.success & jnp.all(jnp.isfinite(solution.value))
        basis = basis.at[:, index + 1].set(_orthogonalize(basis, solution.value))
        return (
            basis,
            success,
            num_steps + solution.num_steps,
            jnp.maximum(max_residual, solution.residual_norm),
        )

    basis, success, num_steps, max_residual = jax.lax.fori_loop(
        0,
        dimension - 1,
        step,
        (basis, jnp.asarray(True), jnp.asarray(0), jnp.asarray(0.0, dtype=seed.dtype)),
    )
    # Project the original operator, not an inverted projected resolvent.
    images = (
        problem.operator.matrix @ basis
        if isinstance(problem.operator, lx.MatrixLinearOperator)
        else jax.lax.map(problem.operator.mv, basis.T).T
    )
    action = _project_basis(
        basis,
        images,
        seed,
        problem.steps,
        exact=dimension == seed.size,
        success=success,
        return_basis=return_basis,
    )
    return action, {"num_steps": num_steps, "max_residual_norm": max_residual}


def _gmres_power_action(
    problem: _PowerProblem,
    dimension: int,
    *,
    rtol: float = 1e-11,
    atol: float = 1e-13,
    max_steps: int = 100,
    restart: int = 50,
    return_basis=False,
):
    """Build a resolvent basis with checked, unpreconditioned operator-based GMRES.

    Private settings permit numerical experiments without exposing solver choice
    on the public method. Statistics count GMRES restart cycles, not matvecs.
    """
    seed = problem.initial
    shift = _rational_shift(problem.steps)
    population_size = seed.size - int(problem.augmented)
    spec = jax.ShapeDtypeStruct((population_size,), seed.dtype)

    def population_action(vector):
        """Apply the population block, removing the affine auxiliary coordinate."""
        if problem.augmented:
            return problem.operator.mv(jnp.concatenate((vector, jnp.zeros(1))))[:-1]
        return problem.operator.mv(vector)

    shifted = lx.FunctionLinearOperator(
        lambda x: (1 + shift) * x - shift * population_action(x), spec
    )
    solver = lx.GMRES(
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        restart=min(restart, population_size),
    )
    state = solver.init(shifted, {})
    source = (
        problem.operator.mv(jnp.zeros_like(seed).at[-1].set(1))[:-1]
        if problem.augmented
        else jnp.zeros_like(seed)
    )

    def solve(vector):
        """Eliminate affine coordinates, solve the population block, and check residuals."""
        # The auxiliary diagonal of the augmented resolvent is one. Eliminate
        # it analytically rather than asking GMRES to resolve the source scale.
        rhs = vector[:-1] + shift * source * vector[-1] if problem.augmented else vector
        solution = lx.linear_solve(shifted, rhs, solver, state=state, throw=False)
        residual = jnp.max(jnp.abs(shifted.mv(solution.value) - rhs))
        tolerance = atol + rtol * jnp.max(jnp.abs(rhs))
        value = (
            jnp.concatenate((solution.value, vector[-1:]))
            if problem.augmented
            else solution.value
        )
        return _ShiftedSolveResult(
            value,
            (solution.result == lx.RESULTS.successful) & (residual <= tolerance),
            solution.stats["num_steps"],
            residual,
        )

    return _rational_projection(problem, dimension, solve, return_basis=return_basis)


def _projected_result(problem, projection, rtol, atol, throw):
    """Check decoded recurrence defects and nested-space differences."""
    fine, coarse = projection.fine, projection.coarse
    defect, projected, coordinates = (
        projection.defect,
        projection.projected,
        projection.coordinates,
    )
    defect, projected, coordinates = jax.tree.map(
        jax.lax.stop_gradient, (defect, projected, coordinates)
    )
    # Batch a few columns to amortize decoding/reduction overhead without
    # materializing every decoded population-sized defect at once.
    weights = jax.lax.map(problem.error_norm, defect.T, batch_size=8)

    def accumulate(_, carry):
        # Weight each column defect by its current reduced-orbit coefficient.
        # This detects leakage missed by a small nested-space difference. Unit
        # propagation does not bound amplification by a general nonnormal A,
        # so the accumulated quantity remains an estimate, not a certificate.
        state, total = carry
        return projected @ state, total + weights @ jnp.abs(state)

    _, defect_estimate = jax.lax.fori_loop(
        0, problem.steps, accumulate, (coordinates, jnp.asarray(0.0))
    )
    defect_estimate *= problem.defect_scale
    result = problem.reconstruct(fine)
    difference = problem.decode(fine - coarse)
    error = jax.lax.stop_gradient(
        jnp.maximum(jnp.sum(jnp.abs(difference)), defect_estimate)
    )
    tolerance = atol + rtol * jnp.sum(jnp.abs(result))
    return _epoch_result(result, error, tolerance, throw, projection.success)


def _polynomial_power_action(
    problem: _PowerProblem, dimension: int, *, return_basis=False
):
    """Build a reorthogonalized Arnoldi basis using only operator applications."""
    seed = problem.initial
    # Powers through k live in K_{k+1}; do not build unused basis vectors.
    dimension = min(dimension, seed.size, problem.steps + 1)
    norm_squared = seed @ seed
    norm = jnp.sqrt(jnp.where(norm_squared > 0, norm_squared, 1.0))
    basis = (
        jnp.zeros((seed.size, dimension), dtype=seed.dtype).at[:, 0].set(seed / norm)
    )

    def step(index, carry):
        """Extend the polynomial space, keeping zero columns after breakdown."""
        basis, images = carry
        candidate = problem.operator.mv(basis[:, index])
        images = images.at[:, index].set(candidate)
        basis = basis.at[:, index + 1].set(_orthogonalize(basis, candidate))
        return basis, images

    basis, images = jax.lax.fori_loop(
        0, dimension - 1, step, (basis, jnp.zeros_like(basis))
    )
    # Reuse Arnoldi's operator applications; only the last column is still needed.
    images = images.at[:, -1].set(problem.operator.mv(basis[:, -1]))
    return _project_basis(
        basis,
        images,
        seed,
        problem.steps,
        exact=dimension == seed.size or dimension > problem.steps,
        success=jnp.asarray(True),
        return_basis=return_basis,
    )


def project(problem, dimension, *, polynomial, return_basis=False, shifted_solve=None):
    """Select an Arnoldi kernel without exposing solver policy to callers."""
    if polynomial:
        return _polynomial_power_action(problem, dimension, return_basis=return_basis)
    if isinstance(problem.operator, lx.MatrixLinearOperator):
        solve = (
            _dense_shifted_solve(problem.operator.matrix, problem.steps)
            if shifted_solve is None
            else shifted_solve
        )
        action, _ = _rational_projection(
            problem, dimension, solve, return_basis=return_basis
        )
        return action
    action, _ = _gmres_power_action(problem, dimension, return_basis=return_basis)
    return action
