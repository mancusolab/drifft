"""Prepared power problems and numerical outcome contracts."""

from collections.abc import Callable
from typing import ClassVar, NamedTuple, Protocol, final

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx


class _KrylovConfig(Protocol):
    """Configuration consumed by projection and sensitivity kernels."""

    dimension: int
    rtol: float
    atol: float
    _polynomial: ClassVar[bool]


class EvolutionStatus(eqx.Enumeration):
    r"""Numerical outcomes; successful power checks do not certify backend accuracy.

    - `successful`: Finite output with accepted power-error and inner-solve checks.
    - `nonfinite`: Nonfinite output or error estimate.
    - `unconverged`: Power-error estimate exceeds the configured tolerance.
    - `linear_solve_failed`: A shifted solve failed its status or residual checks.

    Compare a result's `status` with these members. A failed shifted solve takes
    precedence over nonfiniteness, which takes precedence over power nonconvergence.
    """

    successful = ""
    nonfinite = "Nonfinite evolution output; check input mass and numerical settings."
    unconverged = "Power action did not converge; increase dimension or use Sequential."
    linear_solve_failed = "Shifted linear solve failed; check the reduced operator."


class EpochResult(NamedTuple):
    r"""Evolved mass, power-action L1 error estimate, convergence and status.

    The estimate excludes one-generation approximation and representation error.
    With `throw=False`, use the distribution only after inspecting convergence.

    **Fields:**

    - `distribution`: Float64 full or interior mass, with the input population
      shape. Signed inputs are not normalized.
    - `error_estimate`: Scalar absolute L1 power-error estimate in population
      coordinates; zero for Sequential and identity actions.
    - `converged`: Scalar boolean, true exactly when `status` is successful.
    - `status`: `EvolutionStatus` describing the numerical outcome.
    """

    distribution: jax.Array
    error_estimate: jax.Array
    converged: jax.Array
    status: EvolutionStatus


# Document the generated constructor without replacing its implementation.
EpochResult.__new__.__module__ = __name__
EpochResult.__new__.__doc__ = r"""Construct an immutable result without validation.

**Arguments:**

The fields documented on `EpochResult`, supplied by position or keyword.

**Returns:**

An `EpochResult` holding the supplied values. Normal evolution calls construct this
result automatically; constructing it directly performs no numerical checks.
"""


class EpochProblem(Protocol):
    r"""Read-only algorithm inputs returned by `prepare_epoch`.

    Approximate `operator**steps @ initial` in the provided working coordinates,
    then use `result` to reconstruct population mass and check convergence.
    `steps` is the prepared exponent; reduced representations may absorb an
    initial generation into encoding. Use this exponent rather than the original
    requested generation count. Immigration may add an auxiliary coordinate.

    This protocol describes drifft's prepared PyTree; obtain instances through
    `prepare_epoch`. It is not a constructor or an interface for supplying
    arbitrary user-defined problem objects to built-in methods.
    """

    @property
    def operator(self) -> lx.AbstractLinearOperator:
        r"""Linear one-step action in the prepared working coordinates."""
        ...

    @property
    def initial(self) -> jax.Array:
        r"""Initial working-state vector, including any auxiliary coordinate."""
        ...

    @property
    def steps(self) -> int:
        r"""Nonnegative static exponent for the prepared linear action."""
        ...

    def reconstruct(self, state: jax.Array) -> jax.Array:
        r"""Convert a final working state to full or interior population mass.

        **Arguments:**

        - `state`: Vector matching `initial.shape`.

        **Returns:**

        Population mass including any final source event and affine offset.
        """
        ...

    def error_norm(self, delta: jax.Array) -> jax.Array:
        r"""Measure working-state error in the representation's acceptance norm.

        **Arguments:**

        - `delta`: Error vector matching `initial.shape`.

        **Returns:**

        Scalar absolute L1 error in population coordinates, including any
        auxiliary-coordinate defect. Reconstruction's affine offset is excluded.
        """
        ...

    def result(
        self,
        state: jax.Array,
        *,
        error_estimate: float | jax.Array = 0.0,
        tolerance: float | jax.Array = 0.0,
        throw: bool = True,
    ) -> EpochResult:
        r"""Reconstruct mass and apply the shared finite-output/error checks.

        **Arguments:**

        - `state`: Final working vector, matching `initial.shape`.
        - `error_estimate`: Nonnegative absolute power-error estimate in population
          coordinates; zero for exact iteration. Excludes backend error.
        - `tolerance`: Nonnegative absolute acceptance threshold for that estimate.
        - `throw`: Raise on failure by default; false retains the candidate.

        **Returns:**

        An `EpochResult` with reconstructed mass, estimate, status and convergence.

        **Raises:**

        `ValueError` for incompatible state or scalar shapes. Negative estimates
        or invalid tolerances raise regardless of `throw`.
        Runtime errors for nonfinite mass/estimate or an excessive estimate when
        `throw=True`. A failed batch member may fail the whole batch.
        """
        ...


@final
class _PowerProblem(eqx.Module):
    """A linear power in working coordinates with an affine output reconstruction."""

    operator: lx.AbstractLinearOperator
    initial: jax.Array
    steps: int
    output_map: Callable[[jax.Array], jax.Array]
    output_offset: jax.Array
    augmented: bool
    defect_scale: jax.Array

    def decode(self, state: jax.Array) -> jax.Array:
        """Apply only the linear physical-space output map, excluding the offset."""
        return self.output_map(state[:-1] if self.augmented else state)

    def reconstruct(self, state: jax.Array) -> jax.Array:
        """Recover final population mass, including the last immigration event."""
        return self.decode(state) + self.output_offset

    def error_norm(self, delta: jax.Array) -> jax.Array:
        """Measure physical and auxiliary-coordinate recurrence defects."""
        auxiliary = jnp.abs(delta[-1]) if self.augmented else 0.0
        return jnp.sum(jnp.abs(self.decode(delta))) + auxiliary

    def result(
        self,
        state: jax.Array,
        *,
        error_estimate: float | jax.Array = 0.0,
        tolerance: float | jax.Array = 0.0,
        throw: bool = True,
    ) -> EpochResult:
        """Implement the public prepared-problem result contract."""
        state = jnp.asarray(state, dtype=self.initial.dtype)
        error_estimate = jnp.asarray(error_estimate, dtype=jnp.float64)
        tolerance = jnp.asarray(tolerance, dtype=jnp.float64)
        if state.shape != self.initial.shape:
            raise ValueError("state must match the prepared initial shape")
        if error_estimate.shape != () or tolerance.shape != ():
            raise ValueError("error_estimate and tolerance must be scalars")
        # These are algorithm-domain preconditions. A nonfinite estimate remains
        # a recoverable numerical outcome, while a negative one is invalid input.
        state = eqx.error_if(
            state,
            (error_estimate < 0) | ~jnp.isfinite(tolerance) | (tolerance < 0),
            "error_estimate must be nonnegative; tolerance must be finite and nonnegative",
        )
        return _epoch_result(self.reconstruct(state), error_estimate, tolerance, throw)


def _epoch_inputs(distribution, immigration, gene_copies, generations, segregating):
    """Normalize shared inputs and enforce static shape contracts."""
    if (
        isinstance(generations, bool)
        or not isinstance(generations, int)
        or generations < 0
    ):
        raise ValueError("generations must be a nonnegative Python integer")
    if not isinstance(segregating, bool):
        raise ValueError("segregating must be a Python bool")
    distribution = jnp.asarray(distribution, dtype=jnp.float64)
    size = gene_copies - 1 if segregating else gene_copies + 1
    if distribution.shape != (size,):
        raise ValueError(f"distribution must have {size} entries")
    immigration = (
        jnp.zeros_like(distribution)
        if immigration is None
        else jnp.asarray(immigration, dtype=jnp.float64)
    )
    if immigration.shape != distribution.shape:
        raise ValueError("immigration must have the same shape as distribution")
    return distribution, immigration


def _epoch_result(distribution, error, tolerance, throw, solve_success=True):
    """Apply the finite-output, inner-solve and power-approximation failure policy."""
    finite = jnp.all(jnp.isfinite(distribution)) & jnp.isfinite(error)
    # Apply failure precedence from least to most fundamental: a failed inner
    # solve invalidates the projected action even if its error estimate is small.
    status = EvolutionStatus.where(
        error <= tolerance, EvolutionStatus.successful, EvolutionStatus.unconverged
    )
    status = EvolutionStatus.where(finite, status, EvolutionStatus.nonfinite)
    status = EvolutionStatus.where(
        solve_success, status, EvolutionStatus.linear_solve_failed
    )
    converged = status == EvolutionStatus.successful
    if throw:
        distribution = eqx.error_if(
            distribution,
            ~converged,
            "epoch failed: nonfinite output, failed shifted solve or unconverged power action; "
            "check inputs and increase dimension or use Sequential",
        )
    return EpochResult(distribution, error, converged, status)


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
