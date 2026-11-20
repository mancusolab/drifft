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
from .problem import EpochProblem, _identity


def validate_representation(backend, frequency_map, *, method_name):
    """Validate the interfaces shared by epoch methods."""
    if not isinstance(frequency_map, AbstractFrequencyMap):
        raise TypeError("frequency_map must be an AbstractFrequencyMap")
    if not isinstance(backend, AbstractTransitionBackend):
        raise TypeError(f"{method_name} requires an AbstractTransitionBackend")


def validate_domain(backend, distribution, frequency_map):
    """Retain runtime domain checks even when no power action is scheduled."""
    if isinstance(backend, Jacobi) and backend._supports_reduced_map(frequency_map):
        distribution = backend._validate_parameters(distribution, frequency_map)
    return distribution


@singledispatch
def _prepare_epoch(backend, frequency_map, generations, segregating, source, augmented):
    """Prepare a representation without requiring algorithm methods on every backend."""
    raise TypeError("unsupported epoch backend")


@_prepare_epoch.register(AbstractTransitionBackend)
def _prepare_transition(
    backend, frequency_map, generations, segregating, source, augmented
):
    """Use a one-generation population-space transition as the working operator."""
    transition = Transition(backend, frequency_map)
    operation = partial(transition.advance, segregating=segregating)
    size = source.size + int(augmented)

    def action(state):
        """Advance mass and carry the homogeneous coordinate for source input."""
        if augmented:
            # Homogeneous coordinates turn x_next = A x + source into one
            # linear power: [x; 1] -> [A x + source; 1]. The auxiliary coordinate
            # must be propagated too, including in Krylov residual checks.
            return jnp.concatenate(
                (operation(state[:-1]) + source * state[-1], state[-1:])
            )
        return operation(state)

    operator = lx.FunctionLinearOperator(
        action, jax.ShapeDtypeStruct((size,), source.dtype)
    )
    return EpochProblem(
        operator,
        _identity,
        generations,
        _identity,
        jnp.zeros_like(source),
        augmented,
        1 + generations * jnp.sum(jnp.abs(source)) if augmented else jnp.asarray(1.0),
    )


@_prepare_epoch.register(Jacobi)
def _prepare_jacobi(
    backend, frequency_map, generations, segregating, source, augmented
):
    """Encode the reduced action and affine forcing without choosing a power algorithm."""
    if not backend._supports_reduced_map(frequency_map):
        return _prepare_transition(
            backend, frequency_map, generations, segregating, source, augmented
        )
    source = backend._validate_parameters(source, frequency_map)
    encode = eqx.Partial(
        backend._geometry.encode, frequency_map=frequency_map, segregating=segregating
    )
    # Encoding already applies D.T, the selected one-step evaluation transpose.
    # Thus g generations need (H.T)**(g-1), followed by E.T decoding. In
    # particular, one generation requires no reduced matrix or power at all.
    decode = eqx.Partial(backend._geometry.decode, segregating=segregating)
    if generations == 1:
        operator = lx.IdentityLinearOperator(
            jax.ShapeDtypeStruct(
                (backend.rank + (0 if segregating else 2),), source.dtype
            )
        )
        return EpochProblem(
            operator, encode, 0, decode, source, False, jnp.asarray(1.0)
        )
    encoded_source = None
    if augmented:
        # Earlier source events must pass through the next encoding step. The
        # final event has no subsequent transition and is added in physical
        # coordinates by output_offset, outside the reduced power.
        encoded_source = backend._geometry.encode(source, frequency_map, segregating)
    defect_scale = (
        1 + (generations - 1) * jnp.sum(jnp.abs(source))
        if augmented
        else jnp.asarray(1.0)
    )
    return EpochProblem(
        _JacobiOperator(backend, frequency_map, encoded_source, segregating),
        encode,
        generations - 1,
        decode,
        source,
        augmented,
        defect_scale,
    )
