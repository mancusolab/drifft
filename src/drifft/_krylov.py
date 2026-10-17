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
from .backends import AbstractTransitionBackend


def _power_action_with_status(
    matrix: jax.Array, seed: jax.Array, steps: int, dimension: int
):
    """Project the original matrix onto a twice-reorthogonalized resolvent basis."""
    size = seed.size
    dimension = min(size, dimension)
    shift = steps / 8
    identity = jnp.eye(size)
    operator = lx.MatrixLinearOperator(identity + shift * (identity - matrix))
    solver = lx.LU()
    factor = solver.init(operator, {})
    # Safe normalization also keeps zero and happy-breakdown branches finite in AD.
    norm_squared = seed @ seed
    norm = jnp.sqrt(jnp.where(norm_squared > 0, norm_squared, 1.0))
    basis = jnp.zeros((size, dimension)).at[:, 0].set(seed / norm)

    def step(index, carry):
        """Add a resolvent Arnoldi vector, leaving unused columns zero at breakdown."""
        basis, success = carry
        solution = lx.linear_solve(
            operator, basis[:, index], solver, state=factor, throw=False
        )
        candidate = solution.value
        success = (
            success
            & (solution.result == lx.RESULTS.successful)
            & jnp.all(jnp.isfinite(candidate))
        )
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
    projected = basis.T @ matrix @ basis
    coordinates = basis.T @ seed
    result = basis @ (jnp.linalg.matrix_power(projected, steps) @ coordinates)
    defect = matrix @ basis - basis @ projected
    if dimension == size:
        # A complete basis (or invariant-subspace breakdown) needs no truncation
        # estimate. Finiteness is still checked by the public result contract.
        return result, result, defect, projected, coordinates, success
    coarse_dimension = max(1, dimension // 2)
    coarse = basis[:, :coarse_dimension] @ (
        jnp.linalg.matrix_power(projected[:coarse_dimension, :coarse_dimension], steps)
        @ coordinates[:coarse_dimension]
    )
    return result, coarse, defect, projected, coordinates, success


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
    fine = basis @ (jnp.linalg.matrix_power(projected, problem.steps) @ coordinates)
    defect = images - basis @ projected
    if dimension == seed.size:
        coarse = fine
    else:
        half = max(1, dimension // 2)
        coarse = basis[:, :half] @ (
            jnp.linalg.matrix_power(projected[:half, :half], problem.steps)
            @ coordinates[:half]
        )
    return (fine, coarse, defect, projected, coordinates, success), {
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
    weights = jax.lax.map(problem.error_norm, defect.T)

    def accumulate(_, carry):
        state, total = carry
        return projected @ state, total + weights @ jnp.abs(state)

    _, defect_estimate = jax.lax.fori_loop(
        0, problem.steps, accumulate, (coordinates, jnp.asarray(0.0))
    )
    defect_estimate *= problem.defect_scale
    result = problem.reconstruct(fine)
    difference = problem.decode(fine - coarse)
    error = jnp.maximum(jnp.sum(jnp.abs(difference)), defect_estimate)
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

    Derivatives follow the input-dependent basis and projected power. Derivatives
    at zero seeds or changes in Arnoldi breakdown are not guaranteed. This is not
    a fixed linear operator and does not provide a matched transpose.
    Lineax raises on failed tangent or transpose solves even with `throw=False`.
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
        if isinstance(problem.operator, lx.MatrixLinearOperator):
            action = _power_action_with_status(
                problem.operator.matrix, problem.initial, problem.steps, self.dimension
            )
        else:
            action, _ = _gmres_power_action(problem, self.dimension)
        fine, coarse, defect, projected, coordinates, success = action
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

        Jacobi selection-domain checks apply even when `throw=False`. Power
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


def _polynomial_power_action(problem: _PowerProblem, dimension: int):
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
    fine = basis @ (jnp.linalg.matrix_power(projected, problem.steps) @ coordinates)
    defect = images - basis @ projected
    if dimension == seed.size or dimension > problem.steps:
        # The space can represent the requested polynomial exactly; recurrence
        # defects still detect numerical loss of invariance or orthogonality.
        coarse = fine
    else:
        half = max(1, dimension // 2)
        coarse = basis[:, :half] @ (
            jnp.linalg.matrix_power(projected[:half, :half], problem.steps)
            @ coordinates[:half]
        )
    return fine, coarse, defect, projected, coordinates


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
    backend approximation error. Derivatives follow the input-dependent basis,
    except that zero seeds and changes in Arnoldi breakdown have no accuracy
    guarantee. This method does not provide a matched expectation transpose.
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
        fine, coarse, defect, projected, coordinates = _polynomial_power_action(
            problem, self.dimension
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
        Jacobi's selection-domain restrictions remain active.
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
