"""Prepared power problems and numerical outcome contracts."""

from collections.abc import Callable
from typing import NamedTuple, final

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx


class EvolutionStatus(eqx.Enumeration):
    r"""Numerical outcomes; successful power checks do not certify backend accuracy."""

    successful = ""
    nonfinite = "Nonfinite evolution output; check input mass and numerical settings."
    unconverged = "Power action did not converge; increase dimension or use Sequential."
    linear_solve_failed = "Shifted linear solve failed; check the reduced operator."


class EpochResult(NamedTuple):
    r"""Evolved mass, power-action L1 error estimate, convergence and status.

    The estimate excludes one-generation approximation and representation error.
    With `throw=False`, use the distribution only after inspecting convergence.
    """

    distribution: jax.Array
    error_estimate: jax.Array
    converged: jax.Array
    status: EvolutionStatus


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
