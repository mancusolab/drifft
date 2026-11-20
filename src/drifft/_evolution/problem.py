r"""Carry prepared power problems and distinguish numerical outcomes from AD targets.

A prepared epoch separates population coordinates from the working state used
by a solver. Its reconstruction has the form $p_{\mathrm{out}}=R A^k z_0+e$:
$A$ acts on the working state, $z_0$ is the prepared seed, $R$ decodes it and $e$
is an affine output offset. If the state is augmented for source input, decoding
removes its homogeneous coordinate before returning population mass.

This contract lets a method evaluate a power without knowing how a backend
encoded the population. It also exposes the encoding/reconstruction dependence
to the enclosing chain rule, rather than requiring each solver to reproduce it.
`preparation.py` constructs these problems; `methods.py` evaluates them.

Result and status objects distinguish nonfinite output, inner-solve failure and
power-approximation error. Error norms account for decoded population errors
and, when present, the auxiliary coordinate. These diagnostics describe the
numerical calculation; the action primitives differentiate its mathematical
state output, not status or error estimates. A successful primal calculation
alone does not establish the accuracy of a sensitivity calculation.
"""

from collections.abc import Callable
from typing import NamedTuple, final

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx


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
      coordinates; zero for [`Sequential`][drifft.Sequential] and identity actions.
    - `converged`: Scalar boolean, true exactly when `status` is successful.
    - `status`: [`EvolutionStatus`][drifft.EvolutionStatus] describing the numerical outcome.
    """

    distribution: jax.Array
    error_estimate: jax.Array
    converged: jax.Array
    status: EvolutionStatus


# Document the generated constructor without replacing its implementation.
EpochResult.__new__.__module__ = __name__
EpochResult.__new__.__doc__ = r"""Construct an immutable result without validation.

**Arguments:**

The fields documented on [`EpochResult`][drifft.EpochResult], supplied by position or keyword.

**Returns:**

An [`EpochResult`][drifft.EpochResult] holding the supplied values. Normal evolution calls construct this
result automatically; constructing it directly performs no numerical checks.
"""


class PowerResult(NamedTuple):
    """Working solution, absolute error estimate and acceptance settings.

    The driver compares the error estimate with `atol + rtol *
    norm1(distribution)` after reconstruction. A failed inner solve takes
    precedence over the error estimate.
    """

    state: jax.Array
    error_estimate: float | jax.Array = 0.0
    atol: float | jax.Array = 0.0
    rtol: float | jax.Array = 0.0
    solve_success: bool | jax.Array = True


@final
class EpochProblem(eqx.Module):
    """Reusable epoch encoding, working operator, exponent and reconstruction.

    Obtain a problem from `prepare_epoch`. Encode incoming population mass
    with `encode`, pass the seed to `method.solve`, and finish with `result`.
    A zero working exponent still requires reconstruction.
    """

    operator: lx.AbstractLinearOperator
    input_map: Callable[[jax.Array], jax.Array]
    steps: int
    output_map: Callable[[jax.Array], jax.Array]
    output_offset: jax.Array
    augmented: bool
    defect_scale: jax.Array

    def encode(self, mass: jax.Array) -> jax.Array:
        """Convert population mass to the initial working vector for this hold."""
        mass = jnp.asarray(mass, dtype=jnp.float64)
        if mass.shape != self.output_offset.shape:
            raise ValueError(
                f"distribution must have {self.output_offset.size} entries"
            )
        state = self.input_map(mass)
        if self.augmented:
            state = jnp.concatenate((state, jnp.ones(1, dtype=state.dtype)))
        return state

    def decode(self, state: jax.Array) -> jax.Array:
        """Apply only the linear physical-space output map, excluding the offset."""
        return self.output_map(state[:-1] if self.augmented else state)

    def reconstruct(self, state: jax.Array) -> jax.Array:
        """Recover final population mass, including the final source contribution."""
        return self.decode(state) + self.output_offset

    def error_norm(self, delta: jax.Array) -> jax.Array:
        """Measure physical and auxiliary-coordinate recurrence defects."""
        auxiliary = jnp.abs(delta[-1]) if self.augmented else 0.0
        return jnp.sum(jnp.abs(self.decode(delta))) + auxiliary

    def result(
        self,
        solution: PowerResult,
        *,
        throw: bool = True,
    ) -> EpochResult:
        """Reconstruct the working solution and check its numerical diagnostics."""
        state = jnp.asarray(solution.state, dtype=jnp.float64)
        error_estimate = jnp.asarray(solution.error_estimate, dtype=jnp.float64)
        tolerance = jnp.asarray(solution.atol, dtype=jnp.float64)
        relative = jnp.asarray(solution.rtol, dtype=state.dtype)
        if state.shape != self.operator.in_structure().shape:
            raise ValueError("state must match the prepared working shape")
        if error_estimate.shape != () or tolerance.shape != () or relative.shape != ():
            raise ValueError("error_estimate and tolerance must be scalars")
        # These are algorithm-domain preconditions. A nonfinite estimate remains
        # a recoverable numerical outcome, while a negative one is invalid input.
        state = eqx.error_if(
            state,
            (error_estimate < 0)
            | ~jnp.isfinite(tolerance)
            | (tolerance < 0)
            | ~jnp.isfinite(relative)
            | (relative < 0),
            "error_estimate must be nonnegative; tolerance must be finite and nonnegative",
        )
        distribution = self.reconstruct(state)
        tolerance = jnp.where(
            relative == 0,
            tolerance,
            tolerance + relative * jnp.sum(jnp.abs(distribution)),
        )
        return _epoch_result(
            distribution, error_estimate, tolerance, throw, solution.solve_success
        )


def _epoch_source(source_mass, gene_copies, generations, segregating):
    """Normalize shared inputs and enforce static shape contracts."""
    if (
        isinstance(generations, bool)
        or not isinstance(generations, int)
        or generations < 0
    ):
        raise ValueError("generations must be a nonnegative Python integer")
    if not isinstance(segregating, bool):
        raise ValueError("segregating must be a Python bool")
    size = gene_copies - 1 if segregating else gene_copies + 1
    source_mass = (
        jnp.zeros(size, dtype=jnp.float64)
        if source_mass is None
        else jnp.asarray(source_mass, dtype=jnp.float64)
    )
    if source_mass.shape != (size,):
        raise ValueError("source_mass must have the same shape as distribution")
    return source_mass


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


def _working_problem(operator, steps):
    """Represent an unencoded power action for sensitivity calculations."""
    return EpochProblem(
        operator,
        lambda x: x,
        steps,
        lambda x: x,
        jnp.zeros(operator.in_structure().shape, dtype=jnp.float64),
        False,
        jnp.asarray(1.0),
    )
