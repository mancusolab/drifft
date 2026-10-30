"""Polynomial and rational Krylov power actions on prepared operators."""

# pattern: Functional Core
import math
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from ._epochs import (
    AbstractEpochMethod,
    EpochResult,
    _epoch_result,
    _PowerProblem,
    _prepare_epoch,
)
from ._frequency import AbstractFrequencyMap
from ._jacobi import Jacobi
from ._power import _cached_power_action, _frechet_action
from ._power import _power_action as _operator_power_action
from .backends import AbstractTransitionBackend


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


@jax.custom_jvp
def _require_krylov_derivative(value, supported):
    """Guard direct AD through raw Arnoldi helpers used in research diagnostics.

    Public Krylov methods use the separate mathematical Frechet rule instead.
    """
    return value


@_require_krylov_derivative.defjvp
def _require_krylov_derivative_jvp(primals, tangents):
    """Keep the failure check in both the tangent and transposed tangent paths."""
    value, supported = primals
    tangent, _ = tangents
    # A checked residual multiplier survives transposition to reverse mode.
    # Checking only the primal result can be eliminated when requesting grad.
    scale = eqx.error_if(
        jnp.ones((), dtype=value.dtype),
        ~supported,
        "Krylov derivatives require a nonzero seed and no Arnoldi breakdown; "
        "use Sequential with FFT or NUFFT for derivatives in this regime. "
        "throw=False applies only to forward evolution failures.",
    )
    return _require_krylov_derivative(value, supported), scale * tangent


def _guard_krylov_derivative(value, basis, norm_squared):
    """Reject zero seeds and any zero column left by the breakdown branch."""
    supported = (norm_squared > 0) & jnp.all(jnp.sum(basis * basis, axis=0) > 0)
    return _require_krylov_derivative(value, supported)


def _power_action_with_status(
    matrix: jax.Array,
    seed: jax.Array,
    steps: int,
    dimension: int,
    *,
    return_basis=False,
    shifted_solve=None,
):
    """Project the original matrix onto a twice-reorthogonalized resolvent basis."""
    size = seed.size
    dimension = min(size, dimension)
    shift = steps / 8
    if shifted_solve is None:
        identity = jnp.eye(size)
        operator = lx.MatrixLinearOperator(identity + shift * (identity - matrix))
        solver = lx.LU()
        factor = solver.init(operator, {})
    # Keep primal zero seeds finite; the derivative guard rejects these branches.
    norm_squared = seed @ seed
    norm = jnp.sqrt(jnp.where(norm_squared > 0, norm_squared, 1.0))
    basis = jnp.zeros((size, dimension)).at[:, 0].set(seed / norm)

    def step(index, carry):
        """Add a resolvent Arnoldi vector, leaving unused columns zero at breakdown."""
        basis, success = carry
        if shifted_solve is None:
            solution = lx.linear_solve(
                operator, basis[:, index], solver, state=factor, throw=False
            )
            candidate = solution.value
            solved = solution.result == lx.RESULTS.successful
        else:
            candidate, solved = shifted_solve(basis[:, index])
        success = success & solved & jnp.all(jnp.isfinite(candidate))
        original_squared = candidate @ candidate
        for _ in range(2):
            candidate = candidate - basis @ (basis.T @ candidate)
        squared = candidate @ candidate
        active = squared > (64 * jnp.finfo(jnp.float64).eps) ** 2 * original_squared
        denominator = jnp.sqrt(jnp.where(active, squared, 1.0))
        return basis.at[:, index + 1].set(
            jnp.where(active, candidate / denominator, 0.0)
        ), success

    basis, success = jax.lax.fori_loop(
        0, dimension - 1, step, (basis, jnp.asarray(True))
    )
    # Project A itself: inverting the projected resolvent gives a different,
    # potentially unstable power approximation for this nonnormal operator.
    images = matrix @ basis
    projected = basis.T @ images
    coordinates = basis.T @ seed
    result = basis @ _matrix_power_matvec(projected, coordinates, steps)
    result = _guard_krylov_derivative(result, basis, norm_squared)
    defect = images - basis @ projected
    if dimension == size:
        # A complete basis (or invariant-subspace breakdown) needs no truncation
        # estimate. Finiteness is still checked by the public result contract.
        action = (result, result, defect, projected, coordinates, success)
        return (*action, basis) if return_basis else action
    coarse_dimension = max(1, dimension // 2)
    coarse = basis[:, :coarse_dimension] @ _matrix_power_matvec(
        projected[:coarse_dimension, :coarse_dimension],
        coordinates[:coarse_dimension],
        steps,
    )
    action = (result, coarse, defect, projected, coordinates, success)
    return (*action, basis) if return_basis else action


def _power_action(matrix: jax.Array, seed: jax.Array, steps: int, dimension: int):
    """Compatibility seam for research diagnostics; public methods retain solve status."""
    return _power_action_with_status(matrix, seed, steps, dimension)[:5]


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
    """Build a resolvent basis with checked, unpreconditioned matrix-free GMRES.

    Private settings permit numerical experiments without exposing solver choice
    on the public method. Statistics count GMRES restart cycles, not matvecs.
    """
    seed = problem.initial
    dimension = min(seed.size, dimension)
    shift = problem.steps / 8
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
    norm_squared = seed @ seed
    norm = jnp.sqrt(jnp.where(norm_squared > 0, norm_squared, 1.0))
    basis = (
        jnp.zeros((seed.size, dimension), dtype=seed.dtype).at[:, 0].set(seed / norm)
    )

    def step(index, carry):
        """Solve a shifted system and retain its status before orthogonalization."""
        basis, success, num_steps, max_residual = carry
        vector = basis[:, index]
        rhs = vector[:-1] + shift * source * vector[-1] if problem.augmented else vector
        solution = lx.linear_solve(shifted, rhs, solver, state=state, throw=False)
        residual = jnp.max(jnp.abs(shifted.mv(solution.value) - rhs))
        tolerance = atol + rtol * jnp.max(jnp.abs(rhs))
        success = (
            success
            & (solution.result == lx.RESULTS.successful)
            & jnp.all(jnp.isfinite(solution.value))
            & (residual <= tolerance)
        )
        num_steps += solution.stats["num_steps"]
        max_residual = jnp.maximum(max_residual, residual)
        candidate = (
            jnp.concatenate((solution.value, vector[-1:]))
            if problem.augmented
            else solution.value
        )
        original_squared = candidate @ candidate
        for _ in range(2):
            candidate = candidate - basis @ (basis.T @ candidate)
        squared = candidate @ candidate
        active = squared > (64 * jnp.finfo(seed.dtype).eps) ** 2 * original_squared
        denominator = jnp.sqrt(jnp.where(active, squared, 1.0))
        basis = basis.at[:, index + 1].set(
            jnp.where(active, candidate / denominator, 0.0)
        )
        return basis, success, num_steps, max_residual

    basis, success, num_steps, max_residual = jax.lax.fori_loop(
        0,
        dimension - 1,
        step,
        (basis, jnp.asarray(True), jnp.asarray(0), jnp.asarray(0.0)),
    )
    images = jax.lax.map(problem.operator.mv, basis.T).T
    projected = basis.T @ images
    coordinates = basis.T @ seed
    fine = basis @ _matrix_power_matvec(projected, coordinates, problem.steps)
    fine = _guard_krylov_derivative(fine, basis, norm_squared)
    defect = images - basis @ projected
    if dimension == seed.size:
        coarse = fine
    else:
        half = max(1, dimension // 2)
        coarse = basis[:, :half] @ _matrix_power_matvec(
            projected[:half, :half], coordinates[:half], problem.steps
        )
    action = (fine, coarse, defect, projected, coordinates, success)
    if return_basis:
        action = (*action, basis)
    return action, {
        "num_steps": num_steps,
        "max_residual_norm": max_residual,
    }


def _projected_result(
    problem,
    fine,
    coarse,
    defect,
    projected,
    coordinates,
    rtol,
    atol,
    throw,
    success=True,
):
    """Check decoded recurrence defects and nested-space differences."""
    defect, projected, coordinates = jax.tree.map(
        jax.lax.stop_gradient, (defect, projected, coordinates)
    )
    # Batch a few columns to amortize decoding/reduction overhead without
    # materializing every decoded population-sized defect at once.
    weights = jax.lax.map(problem.error_norm, defect.T, batch_size=8)

    def accumulate(_, carry):
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
    return _epoch_result(result, error, tolerance, throw, success)


@final
class RationalKrylov(AbstractEpochMethod):
    r"""Approximate a prepared power action using a resolvent Krylov basis.

    Each basis vector solves $[I+\tau(I-A)]z=v$, with $\tau=g/8$ for
    working exponent $g$. Jacobi uses a reusable dense LU factorization;
    FFT/NUFFT use unpreconditioned matrix-free Lineax GMRES. Solver choice is
    internal. Inner failures are reported as `linear_solve_failed`.

    GMRES uses rtol=1e-11, atol=1e-13, restart=50 (capped by state size), and
    at most 100 restart cycles per solve, with a recomputed residual check.
    These inner tolerances are independent of the outer power tolerances.
    FFT/NUFFT speedups depend on how many operator applications GMRES needs.

    First-order derivatives approximate the mathematical power sensitivity with
    a coupled Krylov action and a two-sided operator pullback. Arnoldi bases are
    not differentiated. Sensitivity or pullback nonconvergence raises even with
    `throw=False`. Higher-order operator sensitivities are not implemented.
    The input-dependent primal approximation has no matched expectation transpose.
    """

    dimension: int = 64
    rtol: float = 1e-8
    atol: float = 1e-12

    def __check_init__(self) -> None:
        if (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, int)
            or self.dimension < 2
        ):
            raise ValueError("dimension must be a Python integer >= 2")
        # math validates Python configuration at construction; tolerances are
        # not traced model parameters or differentiable numerical inputs.
        if not all(
            math.isfinite(value) and value >= 0 for value in (self.rtol, self.atol)
        ):
            raise ValueError("rtol and atol must be finite and nonnegative")

    def validate(
        self, backend: eqx.Module, frequency_map: AbstractFrequencyMap
    ) -> None:
        """Accept transition geometry or Jacobi with its process restrictions."""
        if not isinstance(frequency_map, AbstractFrequencyMap):
            raise TypeError("frequency_map must be an AbstractFrequencyMap")
        if isinstance(backend, Jacobi):
            backend.validate(frequency_map)
        elif not isinstance(backend, AbstractTransitionBackend):
            raise TypeError("RationalKrylov requires FFT, NUFFT or Jacobi geometry")

    def solve(self, problem: _PowerProblem, *, throw: bool = True) -> EpochResult:
        """Power in working coordinates and estimate error after reconstruction."""
        if problem.steps == 0:
            return _epoch_result(
                problem.reconstruct(problem.initial), jnp.asarray(0.0), self.atol, throw
            )
        arrays, static = eqx.partition(problem, eqx.is_array)
        raw_problem = eqx.combine(jax.tree.map(jax.lax.stop_gradient, arrays), static)
        if isinstance(raw_problem.operator, lx.MatrixLinearOperator):
            action = _power_action_with_status(
                raw_problem.operator.matrix,
                raw_problem.initial,
                raw_problem.steps,
                self.dimension,
            )
        else:
            action, _ = _gmres_power_action(raw_problem, self.dimension)
        fine, coarse, defect, projected, coordinates, success = action
        fine = _cached_power_action(
            self, problem.operator, problem.initial, problem.steps, fine
        )
        return _projected_result(
            problem,
            fine,
            coarse,
            defect,
            projected,
            coordinates,
            self.rtol,
            self.atol,
            throw,
            success,
        )

    def _power_action(self, operator, initial, steps):
        """Evaluate an unencoded power and require its numerical checks to pass."""
        return self.solve(_working_problem(operator, initial, steps)).distribution

    def _power_jvp(self, operator, tangent_operator, initial, tangent_initial, steps):
        """Approximate the mathematical power sensitivity without basis AD."""
        value = _operator_power_action(self, operator, initial, steps)
        if tangent_operator is None:
            tangent = _operator_power_action(self, operator, tangent_initial, steps)
        else:
            tangent = _frechet_action(
                self,
                operator,
                initial,
                tangent_operator.tangent,
                tangent_initial,
                steps,
            )
        return value, tangent

    def _frechet_action(self, operator, perturbation, initial, tangent_initial, steps):
        """Compute the coupled power sensitivity using this method's space."""
        return _frechet_power(
            self, operator, perturbation, initial, tangent_initial, steps
        )

    def _frechet_transpose(self, operator, template, initial, cotangent, steps):
        """Compute the operator pullback in its filtered PyTree representation."""
        return _frechet_operator_pullback(
            self, operator, template, initial, cotangent, steps
        )

    @eqx.filter_jit
    def advance(
        self,
        backend: eqx.Module,
        distribution: jax.Array,
        *,
        frequency_map: AbstractFrequencyMap,
        generations: int,
        segregating: bool = False,
        immigration: jax.Array | None = None,
        throw: bool = True,
    ) -> EpochResult:
        r"""Evolve with the contracts of `AbstractEpochMethod.advance`.

        Jacobi process-domain checks apply even when `throw=False`. Power
        estimates exclude representation, quadrature and assembly error. Increase
        rank and dimension independently when validating a new numerical regime.
        """
        self.validate(backend, frequency_map)
        problem = _prepare_epoch(
            backend, distribution, frequency_map, generations, segregating, immigration
        )
        return self.solve(problem, throw=throw)


RationalKrylov.__init__.__doc__ = r"""Configure the power algorithm without preparing geometry.

**Arguments:**

- `dimension`: Maximum Krylov basis size, an integer at least two; capped by
  the prepared working-state dimension.
- `rtol`, `atol`: Finite nonnegative relative and absolute tolerances for the
  reconstructed power-action error estimate, excluding backend error.

**Returns:**

`None`. Shifted solves use dense LU for Jacobi and matrix-free GMRES for
FFT/NUFFT. Solver selection is internal.

**Raises:**

`ValueError` for invalid dimension or tolerances.
"""


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
        original_squared = candidate @ candidate
        for _ in range(2):
            candidate = candidate - basis @ (basis.T @ candidate)
        squared = candidate @ candidate
        active = squared > (64 * jnp.finfo(seed.dtype).eps) ** 2 * original_squared
        denominator = jnp.sqrt(jnp.where(active, squared, 1.0))
        basis = basis.at[:, index + 1].set(
            jnp.where(active, candidate / denominator, 0.0)
        )
        return basis, images

    basis, images = jax.lax.fori_loop(
        0, dimension - 1, step, (basis, jnp.zeros_like(basis))
    )
    # Reuse Arnoldi's operator applications; only the last column is still needed.
    images = images.at[:, -1].set(problem.operator.mv(basis[:, -1]))
    projected = basis.T @ images
    coordinates = basis.T @ seed
    fine = basis @ _matrix_power_matvec(projected, coordinates, problem.steps)
    fine = _guard_krylov_derivative(fine, basis, norm_squared)
    defect = images - basis @ projected
    if dimension == seed.size or dimension > problem.steps:
        # The space can represent the requested polynomial exactly; recurrence
        # defects still detect numerical loss of invariance or orthogonality.
        coarse = fine
    else:
        half = max(1, dimension // 2)
        coarse = basis[:, :half] @ _matrix_power_matvec(
            projected[:half, :half], coordinates[:half], problem.steps
        )
    action = (fine, coarse, defect, projected, coordinates)
    return (*action, basis) if return_basis else action


@final
class PolynomialKrylov(AbstractEpochMethod):
    r"""Approximate repeated evolution with matrix-free polynomial Arnoldi.

    For a prepared operator $A$, seed $b$, and $g$ steps, build an orthonormal
    basis $V$ for $\mathcal K_m(A,b)=\mathrm{span}(b,Ab,\ldots,A^{m-1}b)$.
    With $H=V^TAV$, approximate $A^g b\approx VH^gV^Tb$. Only operator
    applications are needed; there are no shifted solves or dense population
    matrices. FFT, NUFFT, and Jacobi are supported.

    The basis uses O(n m) storage for working-state size n and dimension m.
    Larger dimensions can be necessary for long intervals; no restart or
    automatic Sequential fallback is performed. Convergence checks exclude
    backend approximation error. First-order derivatives use a coupled Krylov
    sensitivity action and a two-sided operator pullback, without basis AD.
    Sensitivity or pullback nonconvergence raises even with `throw=False`.
    Higher-order operator sensitivities are not implemented. This method does
    not provide a matched expectation transpose of its primal approximation.
    """

    dimension: int = 64
    rtol: float = 1e-8
    atol: float = 1e-12

    def __check_init__(self) -> None:
        """Require a finite error budget and a useful maximum basis dimension."""
        if (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, int)
            or self.dimension < 2
        ):
            raise ValueError("dimension must be a Python integer >= 2")
        # math validates Python configuration at construction; tolerances are
        # not traced model parameters or differentiable numerical inputs.
        if not all(
            math.isfinite(value) and value >= 0 for value in (self.rtol, self.atol)
        ):
            raise ValueError("rtol and atol must be finite and nonnegative")

    def validate(
        self, backend: eqx.Module, frequency_map: AbstractFrequencyMap
    ) -> None:
        """Accept transition geometry or Jacobi with its restricted process domain."""
        if not isinstance(frequency_map, AbstractFrequencyMap):
            raise TypeError("frequency_map must be an AbstractFrequencyMap")
        if isinstance(backend, Jacobi):
            backend.validate(frequency_map)
        elif not isinstance(backend, AbstractTransitionBackend):
            raise TypeError("PolynomialKrylov requires FFT, NUFFT or Jacobi geometry")

    def solve(self, problem: _PowerProblem, *, throw: bool = True) -> EpochResult:
        """Power an implicit operator and report reconstructed error diagnostics."""
        if problem.steps == 0:
            return _epoch_result(
                problem.reconstruct(problem.initial), jnp.asarray(0.0), self.atol, throw
            )
        arrays, static = eqx.partition(problem, eqx.is_array)
        raw_problem = eqx.combine(jax.tree.map(jax.lax.stop_gradient, arrays), static)
        fine, coarse, defect, projected, coordinates = _polynomial_power_action(
            raw_problem, self.dimension
        )
        fine = _cached_power_action(
            self, problem.operator, problem.initial, problem.steps, fine
        )
        return _projected_result(
            problem,
            fine,
            coarse,
            defect,
            projected,
            coordinates,
            self.rtol,
            self.atol,
            throw,
        )

    def _power_action(self, operator, initial, steps):
        """Evaluate an unencoded power and require its numerical checks to pass."""
        return self.solve(_working_problem(operator, initial, steps)).distribution

    def _power_jvp(self, operator, tangent_operator, initial, tangent_initial, steps):
        """Approximate the mathematical power sensitivity without basis AD."""
        value = _operator_power_action(self, operator, initial, steps)
        if tangent_operator is None:
            tangent = _operator_power_action(self, operator, tangent_initial, steps)
        else:
            tangent = _frechet_action(
                self,
                operator,
                initial,
                tangent_operator.tangent,
                tangent_initial,
                steps,
            )
        return value, tangent

    def _frechet_action(self, operator, perturbation, initial, tangent_initial, steps):
        """Compute the coupled power sensitivity using this method's space."""
        return _frechet_power(
            self, operator, perturbation, initial, tangent_initial, steps
        )

    def _frechet_transpose(self, operator, template, initial, cotangent, steps):
        """Compute the operator pullback in its filtered PyTree representation."""
        return _frechet_operator_pullback(
            self, operator, template, initial, cotangent, steps
        )

    @eqx.filter_jit
    def advance(
        self,
        backend: eqx.Module,
        distribution: jax.Array,
        *,
        frequency_map: AbstractFrequencyMap,
        generations: int,
        segregating: bool = False,
        immigration: jax.Array | None = None,
        throw: bool = True,
    ) -> EpochResult:
        r"""Evolve with the contracts of `AbstractEpochMethod.advance`.

        Immigration is included through an augmented operator. Inspect convergence
        with `throw=False`, or increase dimension when the power check fails.
        Jacobi's process-domain checks remain active, including with `throw=False`.
        """
        self.validate(backend, frequency_map)
        problem = _prepare_epoch(
            backend, distribution, frequency_map, generations, segregating, immigration
        )
        return self.solve(problem, throw=throw)


PolynomialKrylov.__init__.__doc__ = r"""Configure polynomial powering independently of geometry.

**Arguments:**

- `dimension`: Maximum basis size, a Python integer at least two. Capped by
  the working-state size and the number of steps plus one.
- `rtol`, `atol`: Finite nonnegative relative and absolute tolerances for the
  power-action error estimate; these do not control backend approximation.

**Returns:**

`None`. Supply FFT, NUFFT or Jacobi geometry separately.

**Raises:**

`ValueError` for invalid dimension or tolerances.
"""


def _working_problem(operator, initial, steps):
    """Represent an unencoded power action for sensitivity calculations."""
    return _PowerProblem(
        operator,
        initial,
        steps,
        lambda x: x,
        jnp.zeros_like(initial),
        False,
        jnp.asarray(1.0),
    )


def _sensitivity_projection(
    method, operator, initial, steps, *, dimension=None, shifted_solve=None
):
    """Reuse the method's Arnoldi space, retaining its basis for adjoint factors."""
    problem = _working_problem(operator, initial, steps)
    dimension = method.dimension if dimension is None else dimension
    if isinstance(method, PolynomialKrylov):
        fine, coarse, defect, projected, coordinates, basis = _polynomial_power_action(
            problem, dimension, return_basis=True
        )
        success = jnp.asarray(True)
    elif isinstance(operator, lx.MatrixLinearOperator):
        fine, coarse, defect, projected, coordinates, success, basis = (
            _power_action_with_status(
                operator.matrix,
                initial,
                steps,
                dimension,
                return_basis=True,
                shifted_solve=shifted_solve,
            )
        )
    else:
        action, _ = _gmres_power_action(problem, dimension, return_basis=True)
        fine, coarse, defect, projected, coordinates, success, basis = action
    return fine, coarse, defect, projected, coordinates, success, basis


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
        block = lx.MatrixLinearOperator(jnp.block([[a, e], [jnp.zeros_like(a), a]]))
        if isinstance(method, RationalKrylov):
            shift = steps / 8
            shifted = lx.MatrixLinearOperator(
                jnp.eye(size) + shift * (jnp.eye(size) - a)
            )
            solver = lx.LU()
            factor = solver.init(shifted, {})

            def shifted_solve(rhs):
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
    fine, coarse, defect, projected, coordinates, success, _ = _sensitivity_projection(
        method,
        block,
        seed,
        steps,
        dimension=2 * method.dimension,
        shifted_solve=shifted_solve,
    )
    # Require the combined error to satisfy each output block's tolerance. A large
    # primal must not mask an inaccurate, small sensitivity. This remains an estimate.
    tolerance = jnp.minimum(
        method.atol + method.rtol * jnp.sum(jnp.abs(fine[:size])),
        method.atol + method.rtol * jnp.sum(jnp.abs(fine[size:])),
    )
    result = _projected_result(
        _working_problem(block, seed, steps),
        fine,
        coarse,
        defect,
        projected,
        coordinates,
        0.0,
        tolerance,
        False,
        success,
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
    _, _, left_defect, h, c, left_ok, u = _sensitivity_projection(
        method, operator.T, cotangent, steps
    )
    _, _, right_defect, j, x, right_ok, v = _sensitivity_projection(
        method, operator, initial, steps
    )
    core = _frechet_core(h, j, c, x, steps)
    full = u.shape[1] == initial.size and v.shape[1] == initial.size
    exact_polynomial = (
        isinstance(method, PolynomialKrylov) and min(u.shape[1], v.shape[1]) >= steps
    )
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
    left_weights = jnp.linalg.norm(left_defect, axis=0)
    right_weights = jnp.linalg.norm(right_defect, axis=0)

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
        left_ok
        & right_ok
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
