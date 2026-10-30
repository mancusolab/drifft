"""Representation-independent contracts for constant-size forward evolution."""

# pattern: Functional Core
from abc import abstractmethod
from collections.abc import Callable
from functools import singledispatch
from typing import NamedTuple, final

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from ._frequency import AbstractFrequencyMap, Neutral, Underdominance
from ._operator import Transition
from ._power import _power_action
from .backends import AbstractTransitionBackend, Jacobi
from .backends.jacobi import _assemble, _assemble_diploid


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


class AbstractEpochMethod(eqx.Module):
    r"""An algorithm for a prepared linear power action, independent of geometry."""

    @abstractmethod
    def validate(
        self, backend: eqx.Module, frequency_map: AbstractFrequencyMap
    ) -> None:
        """Reject unsupported representation and process combinations before execution."""
        ...

    @abstractmethod
    def solve(self, problem: _PowerProblem, *, throw: bool = True) -> EpochResult:
        """Evaluate a prepared problem; preparation is internal to drifft."""
        ...

    @abstractmethod
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
        r"""Bind geometry and evolve a constant-size interval.

        **Arguments:**

        - `backend`: Prepared geometry compatible with this method.
        - `distribution`: Real mass, converted to float64; full or interior counts.
        - `frequency_map`: Process shared by every generation.
        - `generations`: Nonnegative Python integer, static under JIT.
        - `segregating`: Discard endpoint mass at each generation.
        - `immigration`: Mass added after every generation, matching the input shape.
        - `throw`: Raise on numerical failure; under vmap this may fail the whole batch.

        **Returns:**

        `EpochResult`; zero generations preserves the input without immigration.

        **Raises:**

        `TypeError` for unsupported combinations; `ValueError` for static input errors.
        Runtime domain checks remain active even with `throw=False`.
        """
        ...


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


@singledispatch
def _prepare_epoch(
    backend, distribution, frequency_map, generations, segregating, immigration
):
    """Prepare a representation without requiring algorithm methods on every backend."""
    raise TypeError("unsupported epoch backend")


def _identity(value):
    return value


@_prepare_epoch.register(AbstractTransitionBackend)
def _prepare_transition(
    backend, distribution, frequency_map, generations, segregating, immigration
):
    distribution, source = _epoch_inputs(
        distribution, immigration, backend.gene_copies, generations, segregating
    )
    transition = Transition(backend, frequency_map)
    operation = transition.advance_segregating if segregating else transition.advance
    augmented = immigration is not None and generations > 0
    initial = (
        jnp.concatenate((distribution, jnp.ones(1))) if augmented else distribution
    )

    def action(state):
        if augmented:
            return jnp.concatenate(
                (operation(state[:-1]) + source * state[-1], state[-1:])
            )
        return operation(state)

    operator = lx.FunctionLinearOperator(
        action, jax.ShapeDtypeStruct(initial.shape, initial.dtype)
    )
    return _PowerProblem(
        operator,
        initial,
        generations,
        _identity,
        jnp.zeros_like(distribution),
        augmented,
        1 + generations * jnp.sum(jnp.abs(source)) if augmented else jnp.asarray(1.0),
    )


@_prepare_epoch.register(Jacobi)
def _prepare_jacobi(
    backend, distribution, frequency_map, generations, segregating, immigration
):
    """Encode the reduced action and affine forcing without choosing a power algorithm."""
    distribution, source = _epoch_inputs(
        distribution, immigration, backend.gene_copies, generations, segregating
    )
    distribution, strength = backend._validate_mass(distribution, frequency_map)
    if generations == 0:
        operator = lx.IdentityLinearOperator(
            jax.ShapeDtypeStruct(distribution.shape, distribution.dtype)
        )
        return _PowerProblem(
            operator,
            distribution,
            0,
            _identity,
            jnp.zeros_like(distribution),
            False,
            jnp.asarray(1.0),
        )
    seed = backend._geometry.encode(distribution, frequency_map, segregating)
    decode = eqx.Partial(backend._geometry.decode, segregating=segregating)
    if generations == 1:
        operator = lx.IdentityLinearOperator(
            jax.ShapeDtypeStruct(seed.shape, seed.dtype)
        )
        return _PowerProblem(operator, seed, 0, decode, source, False, jnp.asarray(1.0))
    if isinstance(frequency_map, (Neutral, Underdominance)):
        matrix, forcing = _assemble(
            strength, backend.gene_copies, backend.rank, backend.padded_rank
        )
        if not segregating:
            extended = jnp.zeros((backend.rank + 2, backend.rank + 2))
            extended = extended.at[:2, :2].set(jnp.eye(2))
            extended = extended.at[2:, 1].set(forcing)
            matrix = extended.at[2:, 2:].set(matrix)
    else:
        matrix = _assemble_diploid(
            frequency_map,
            backend.gene_copies,
            backend.rank,
            backend._diploid_padded_rank,
        )
        if segregating:
            matrix = (
                matrix[2:, 2:] + backend._geometry.endpoint_correction @ matrix[:2, 2:]
            )
    matrix = matrix.T
    augmented = immigration is not None
    if augmented:
        encoded_source = backend._geometry.encode(source, frequency_map, segregating)
        extended = jnp.zeros((seed.size + 1, seed.size + 1))
        extended = extended.at[:-1, :-1].set(matrix)
        extended = extended.at[:-1, -1].set(encoded_source)
        matrix = extended.at[-1, -1].set(1)
        seed = jnp.concatenate((seed, jnp.ones(1)))
    defect_scale = (
        1 + (generations - 1) * jnp.sum(jnp.abs(source))
        if augmented
        else jnp.asarray(1.0)
    )
    return _PowerProblem(
        lx.MatrixLinearOperator(matrix),
        seed,
        generations - 1,
        decode,
        source,
        augmented,
        defect_scale,
    )


def _prepare_evolution(
    method, backend, distribution, frequency_map, generations, segregating, immigration
):
    """Use the common transition for Sequential and backend reductions for powers."""
    if isinstance(method, Sequential):
        return _prepare_transition(
            backend, distribution, frequency_map, generations, segregating, immigration
        )
    return _prepare_epoch(
        backend, distribution, frequency_map, generations, segregating, immigration
    )


@final
class Sequential(AbstractEpochMethod):
    r"""Repeat one-generation transitions without an additional power approximation.

    Construct as `Sequential()`. Geometry is supplied separately to `advance` or
    through `DemographicTransition`. The reported power-action error is zero.
    """

    def validate(
        self, backend: eqx.Module, frequency_map: AbstractFrequencyMap
    ) -> None:
        """Require a one-generation transition backend and frequency map."""
        if not isinstance(backend, AbstractTransitionBackend):
            raise TypeError("Sequential requires an AbstractTransitionBackend")
        if not isinstance(frequency_map, AbstractFrequencyMap):
            raise TypeError("frequency_map must be an AbstractFrequencyMap")

    def solve(self, problem: _PowerProblem, *, throw: bool = True) -> EpochResult:
        """Iterate in the prepared problem's working coordinates."""
        state = _power_action(self, problem.operator, problem.initial, problem.steps)
        return _epoch_result(problem.reconstruct(state), jnp.asarray(0.0), 0.0, throw)

    def _power_action(self, operator, initial, steps):
        """Apply repeated operator actions in working coordinates."""
        return jax.lax.fori_loop(0, steps, lambda _, state: operator.mv(state), initial)

    def _power_jvp(self, operator, tangent_operator, initial, tangent_initial, steps):
        """Evolve the state and its sensitivity without differentiating the loop."""
        if tangent_operator is None:
            return (
                _power_action(self, operator, initial, steps),
                _power_action(self, operator, tangent_initial, steps),
            )

        def step(_, carry):
            state, tangent = carry
            following = operator.mv(state)
            tangent = jax.tree.map(
                jnp.add, operator.mv(tangent), tangent_operator.mv(state)
            )
            return following, tangent

        return jax.lax.fori_loop(0, steps, step, (initial, tangent_initial))

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
        r"""Evolve with the contracts of `AbstractEpochMethod.advance`."""
        self.validate(backend, frequency_map)
        problem = _prepare_evolution(
            self,
            backend,
            distribution,
            frequency_map,
            generations,
            segregating,
            immigration,
        )
        return self.solve(problem, throw=throw)


Sequential.__init__.__doc__ = r"""Configure repeated one-generation evolution.

**Arguments:**

No arguments. Supply geometry separately to `advance` or `DemographicTransition`.

**Returns:**

`None`.
"""
