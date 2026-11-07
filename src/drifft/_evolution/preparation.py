r"""Express population evolution as a power in a backend's working coordinates.

An epoch can be written as $p_{\mathrm{out}}=R A^k z_0+e$, where $z_0$ is the
encoded initial state, $R$ decodes a working state and $e$ is an output offset.
The working state can include a homogeneous scalar coordinate, making repeated
source addition part of a linear power. Preparation defines these maps without
choosing how the power will be evaluated.

This boundary matters for differentiation: encoding, the working operator and
source terms can all depend on process parameters. Keeping those dependencies
in the prepared problem lets ordinary AD compose them with the mathematical
power rules in `autodiff.py`. Backward epoch expectation transposes the affine
preparation/reconstruction in `_demography/actions.py`, including its offsets.

Generic backends use a one-generation [`Transition`][drifft.Transition] action.
Jacobi uses a reduced representation: its encoding already includes one selected
transition, so an epoch of $g>0$ generations requires only $g-1$ reduced powers.
Earlier source events enter the encoded state, while the final source event is
an output offset. Zero- and one-generation cases therefore need separate
preparations, not an artificial extra power. `problem.py` carries this contract;
`methods.py` consumes it to evaluate and reconstruct the result.
"""

# pattern: Functional Core
from functools import partial, singledispatch

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from .._backend import AbstractTransitionBackend, Jacobi
from .._frequency import AbstractFrequencyMap
from .._operator import Transition
from .parameterized import _JacobiOperator
from .problem import _epoch_inputs, _PowerProblem


def validate_representation(backend, frequency_map, *, reduced, method_name):
    """Check configuration, applying assembly restrictions only to reduced powers."""
    if not isinstance(frequency_map, AbstractFrequencyMap):
        raise TypeError("frequency_map must be an AbstractFrequencyMap")
    if not isinstance(backend, AbstractTransitionBackend):
        raise TypeError(f"{method_name} requires an AbstractTransitionBackend")
    if reduced and isinstance(backend, Jacobi):
        backend._validate_reduced_map(frequency_map)


def validate_domain(backend, distribution, frequency_map, *, reduced):
    """Retain runtime domain checks even when no power action is scheduled."""
    if reduced and isinstance(backend, Jacobi):
        distribution = backend._validate_parameters(distribution, frequency_map)
    return distribution


@singledispatch
def _prepare_epoch(
    backend, distribution, frequency_map, generations, segregating, source_mass
):
    """Prepare a representation without requiring algorithm methods on every backend."""
    raise TypeError("unsupported epoch backend")


def _identity(value):
    return value


@_prepare_epoch.register(AbstractTransitionBackend)
def _prepare_transition(
    backend, distribution, frequency_map, generations, segregating, source_mass
):
    distribution, source = _epoch_inputs(
        distribution, source_mass, backend._gene_copies, generations, segregating
    )
    transition = Transition(backend, frequency_map)
    operation = partial(transition.advance, segregating=segregating)
    augmented = source_mass is not None and generations > 0
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
    backend, distribution, frequency_map, generations, segregating, source_mass
):
    """Encode the reduced action and affine forcing without choosing a power algorithm."""
    distribution, source = _epoch_inputs(
        distribution, source_mass, backend._gene_copies, generations, segregating
    )
    distribution = backend._validate_parameters(distribution, frequency_map)
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
    encoded_source = None
    augmented = source_mass is not None
    if augmented:
        # Earlier source events must pass through the next encoding step. The
        # final event has no subsequent transition and is added in physical
        # coordinates by output_offset, outside the reduced power.
        encoded_source = backend._geometry.encode(source, frequency_map, segregating)
        seed = jnp.concatenate((seed, jnp.ones(1)))
    defect_scale = (
        1 + (generations - 1) * jnp.sum(jnp.abs(source))
        if augmented
        else jnp.asarray(1.0)
    )
    return _PowerProblem(
        _JacobiOperator(backend, frequency_map, encoded_source, segregating),
        seed,
        generations - 1,
        decode,
        source,
        augmented,
        defect_scale,
    )
