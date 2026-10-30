"""Adapt transition representations to linear power problems."""

from functools import singledispatch

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from .._frequency import AbstractFrequencyMap, Neutral, Underdominance
from .._operator import Transition
from ..backends import AbstractTransitionBackend, Jacobi
from ..backends.jacobi import _assemble, _assemble_diploid
from .problem import _epoch_inputs, _PowerProblem


def validate_representation(backend, frequency_map, *, reduced, method_name):
    """Check configuration, applying assembly restrictions only to reduced powers."""
    if not isinstance(frequency_map, AbstractFrequencyMap):
        raise TypeError("frequency_map must be an AbstractFrequencyMap")
    if not isinstance(backend, AbstractTransitionBackend):
        raise TypeError(f"{method_name} requires an AbstractTransitionBackend")
    if reduced and isinstance(backend, Jacobi):
        backend.validate(frequency_map)


def validate_domain(backend, distribution, frequency_map, *, reduced):
    """Retain runtime domain checks even when no power action is scheduled."""
    if reduced and isinstance(backend, Jacobi):
        distribution, _ = backend._validate_mass(distribution, frequency_map)
    return distribution


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
            # Homogeneous coordinates turn x_next = A x + source into one
            # linear power: [x; 1] -> [A x + source; 1]. The auxiliary coordinate
            # must be propagated too, including in Krylov residual checks.
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
    # Encoding already applies D.T, the selected one-step evaluation transpose.
    # Thus g generations need (H.T)**(g-1), followed by E.T decoding. In
    # particular, one generation requires no reduced matrix or power at all.
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
            # Recurrent mutation can produce nonzero endpoint coefficients.
            # Fold their interior contribution into E_int D_int before discarding
            # endpoints; simply slicing the full operator would lose that term.
            matrix = (
                matrix[2:, 2:] + backend._geometry.endpoint_correction @ matrix[:2, 2:]
            )
    matrix = matrix.T
    augmented = immigration is not None
    if augmented:
        # Earlier source events must pass through the next encoding step. The
        # final event has no subsequent transition and is added in physical
        # coordinates by output_offset, outside the reduced power.
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
