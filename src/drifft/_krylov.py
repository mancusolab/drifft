"""Rational Krylov power actions, independent of population representation."""

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


@final
class RationalKrylov(AbstractEpochMethod):
    r"""Approximate a prepared power action using a resolvent Krylov basis.

    Currently supports Jacobi geometry with an internal dense LU shifted solve.
    Derivatives follow the input-dependent basis and projected power. Derivatives
    at zero seeds or changes in Arnoldi breakdown are not guaranteed. This is not
    a fixed linear operator and does not provide a matched transpose.
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
        """Require a backend with the validated dense reduced solve strategy."""
        if not isinstance(backend, Jacobi):
            raise TypeError("RationalKrylov currently requires Jacobi geometry")
        backend.validate(frequency_map)

    def solve(self, problem: _PowerProblem, *, throw: bool = True) -> EpochResult:
        """Power in working coordinates and estimate error after reconstruction."""
        if problem.steps == 0:
            return _epoch_result(
                problem.reconstruct(problem.initial), jnp.asarray(0.0), self.atol, throw
            )
        if not isinstance(problem.operator, lx.MatrixLinearOperator):
            raise TypeError(
                "RationalKrylov requires an explicitly reduced matrix; no automatic materialization"
            )
        fine, coarse, defect, projected, coordinates, success = (
            _power_action_with_status(
                problem.operator.matrix, problem.initial, problem.steps, self.dimension
            )
        )
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
        tolerance = self.atol + self.rtol * jnp.sum(jnp.abs(result))
        return _epoch_result(result, error, tolerance, throw, success)

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

`None`. The internal shifted solver is dense LU; it is not a user configuration.

**Raises:**

`ValueError` for invalid dimension or tolerances.
"""
