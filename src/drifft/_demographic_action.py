"""Mathematical demographic actions and their shared HiJAX differentiation rules."""

# pattern: Functional Core
from functools import partial
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.experimental import hijax

from ._evolution.autodiff import (
    _accumulate,
    _has_accum,
    _has_tangent,
    _is_ad_leaf,
    _tangent_or_none,
)
from ._evolution.methods import prepare_epoch
from ._evolution.preparation import validate_domain
from ._evolution.problem import _working_problem


def _stop_arrays(tree):
    """Keep primal diagnostics and adaptive numerical construction outside AD."""
    arrays, static = eqx.partition(tree, eqx.is_array)
    return eqx.combine(jax.tree.map(jax.lax.stop_gradient, arrays), static)


def _expect_epoch(model, backend, values, generations, rate, segregating):
    """Transpose a prepared affine epoch, including its source reward.

    Write preparation as initial = B mass + d and reconstruction as R state + e.
    The backward values are B.T (A.T)**k R.T values; the offset is
    d.T (A.T)**k R.T values + e.T values. Preparing zero mass extracts d.
    This also handles Jacobi's reduced exponent and final-generation source.
    """
    zero = jnp.zeros_like(values)
    source = zero.at[0].set(2 * backend.pop_size * rate) if segregating else None

    def prepare(mass):
        return prepare_epoch(
            model.method,
            backend,
            mass,
            frequency_map=model.frequency_map,
            generations=generations,
            segregating=segregating,
            source_mass=source,
        )

    problem = prepare(zero)
    # Differentiate only the linear coordinate maps, never an adaptive power.
    # Their dependence on process parameters remains available to outer AD.
    seed, encode_transpose = jax.vjp(lambda mass: prepare(mass).initial, zero)
    _, decode_transpose = jax.vjp(problem.decode, jnp.zeros_like(problem.initial))
    terminal = decode_transpose(values)[0]
    working = model.method.solve(
        _working_problem(problem.operator.T, terminal, problem.steps)
    ).distribution
    offset = seed @ working + problem.output_offset @ values
    return encode_transpose(working)[0], offset


def _evaluate(model, state, rate, *, segregating, transpose):
    """Numerically apply the augmented history or its transpose.

    Here H is the represented expectation operator. The flag refers to H:
    transpose=True applies the augmented forward action
    (v, h) -> (H.T v + c h, h); transpose=False applies its transpose
    (f, a) -> (H f, a + c.T f). The scalar coordinate is arbitrary internally,
    including in pullbacks; public advance/expect supply one/zero respectively.
    Epoch power primitives supply mathematical power derivatives; adaptive
    projection machinery is never differentiated by this composition.
    """
    vector, scalar = state
    if transpose:
        result = model._advance(
            vector, segregating=segregating, singleton_rate=rate * scalar, throw=True
        )
        return result.distribution, scalar

    values, offset = vector, scalar
    for backend in model.geometry._backends.values():
        values = validate_domain(
            backend,
            values,
            model.frequency_map,
            reduced=model.method.allows_reduced_representation,
        )
    history = model.demography
    for source, destination, duration in reversed(
        tuple(
            zip(
                history.pop_sizes[:-1],
                history.pop_sizes[1:],
                history.generations,
                strict=True,
            )
        )
    ):
        if segregating:
            offset = offset + 2 * destination * rate * values[0]
        values = model._transition(source, destination).expect(
            values, segregating=segregating
        )
        if duration > 1:
            values, contribution = _expect_epoch(
                model,
                model.geometry._backends[source],
                values,
                duration - 1,
                rate,
                segregating,
            )
            offset = offset + contribution
    return values, offset


def _directional(model, state, rate, direction, *, segregating, transpose):
    """Evaluate D(A(theta) x)[dtheta, dx] through mathematical epoch operations."""
    action = partial(_evaluate, segregating=segregating, transpose=transpose)
    primal, tangent = eqx.filter_jvp(action, (model, state, rate), direction)
    return jax.tree.map(
        lambda value, diff: jnp.zeros_like(value) if diff is None else diff,
        primal,
        tangent,
        is_leaf=lambda x: x is None,
    )


def _parameter_pullback(
    model, state, rate, cotangent, templates, *, segregating, transpose
):
    """Differentiate y.T A(theta) x with state and cotangent fixed.

    Swapping their roles handles the backward action with the same rule.
    The mask selects only active process/rate leaves; geometry and method
    configuration remain fixed. Epoch primitives evaluate the power pullbacks.
    """
    mask = jax.tree.map(
        lambda value: isinstance(value, hijax.GradAccum),
        templates,
        is_leaf=lambda x: x is None or _is_ad_leaf(x),
    )
    active, fixed = eqx.partition((model, rate), mask)
    # <a, A.T b> = <A a, b>: use the forward evaluator for both pairings.
    seed, reward = (state, cotangent) if transpose else (cotangent, state)

    def forward(parameters):
        current_model, current_rate = eqx.combine(parameters, fixed)
        return _evaluate(
            current_model, seed, current_rate, segregating=segregating, transpose=True
        )

    _, pullback = eqx.filter_vjp(forward, active)
    return pullback(reward)[0]


class _DemographicAction(hijax.HiPrim):
    """An oriented augmented history action with solver-independent AD semantics.

    Cached values retain the checked primal and carry no derivative. The JVP
    is attached to the original process, rate, and vector arguments. Geometry
    and numerical method settings cannot participate in differentiation.
    """

    def __init__(self, avals, static, *, segregating, transpose, cached):
        """Declare the orientation-dependent population grid and scalar coordinate."""
        self.in_avals = avals
        model = eqx.combine(avals[0], static[0])
        size = (
            model.demography.final_pop_size
            if transpose
            else model.demography.initial_pop_size
        )
        self.out_aval = (
            ShapedArray((2 * size + (-1 if segregating else 1),), jnp.float64),
            ShapedArray((), jnp.float64),
        )
        self.params = dict(
            static=static, segregating=segregating, orientation=transpose, cached=cached
        )
        super().__init__()

    def expand(self, *args):
        """Evaluate the operator using the configured numerical method."""
        model, state, rate, *cached = eqx.combine(
            args, cast(tuple, self.params["static"])
        )
        if self.params["cached"]:
            return cached[0]
        return _evaluate(model, state, rate, **self._config())

    def _config(self):
        return dict(
            segregating=self.params["segregating"], transpose=self.params["orientation"]
        )

    def jvp(self, primals, tangents):
        """Attach the common derivative of A(theta) x, ignoring cached-value tangents."""
        model, state, rate, *_ = eqx.combine(
            primals, cast(tuple, self.params["static"])
        )
        t_model, t_state, t_rate, *_ = tangents
        if _has_tangent((t_model.geometry, t_model.method)):
            raise TypeError(
                "demographic geometry and numerical method configuration are nondifferentiable"
            )
        direction = jax.tree.map(
            _tangent_or_none, (t_model, t_state, t_rate), is_leaf=_is_ad_leaf
        )
        tangent = _bind_linearized(model, state, rate, direction, **self._config())
        return self(*primals), tangent

    lin, linearized = hijax.linearize_from_jvp
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

    def transpose(self, out_ct, *args):
        """Transpose the vector input by invoking the opposite history action."""
        model, state, rate, *_ = eqx.combine(args, cast(tuple, self.params["static"]))
        if self.params["cached"] or _has_accum((model, rate)):
            raise ValueError(
                "demographic actions are linear only in the augmented vector"
            )
        if _has_accum(state):
            cotangent = jax.tree.map(
                hijax.instantiate_zeros, out_ct, is_leaf=_is_ad_leaf
            )
            _accumulate(
                state,
                _bind_action(
                    model,
                    cotangent,
                    rate,
                    segregating=self.params["segregating"],
                    transpose=not self.params["orientation"],
                ),
            )

    def batch_dim_rule(self, axis_data, dims):
        """Map both output coordinates when any mathematical input is mapped."""
        dim = 0 if any(d is not None for d in jax.tree.leaves(dims)) else None
        return dim, dim


class _DemographicLinearizedAction(hijax.HiPrim):
    """Linearized history action with explicit vector and parameter pullbacks."""

    def __init__(self, avals, static, *, segregating, transpose):
        """Use the same oriented output grid as the primal action."""
        self.in_avals = avals
        action = _DemographicAction(
            avals[:3],
            static[:3],
            segregating=segregating,
            transpose=transpose,
            cached=False,
        )
        self.out_aval = action.out_aval
        self.params = dict(
            static=static, segregating=segregating, orientation=transpose
        )
        super().__init__()

    def _config(self):
        return dict(
            segregating=self.params["segregating"], transpose=self.params["orientation"]
        )

    def expand(self, *args):
        """Evaluate the shared directional identity with epoch numerical kernels."""
        model, state, rate, direction = eqx.combine(
            args, cast(tuple, self.params["static"])
        )
        return _directional(model, state, rate, direction, **self._config())

    def jvp(self, primals, tangents):
        """Retain higher derivatives where the numerical sensitivity kernels support them."""
        inputs = eqx.combine(primals, cast(tuple, self.params["static"]))
        directions = jax.tree.map(_tangent_or_none, tangents, is_leaf=_is_ad_leaf)
        primal, tangent = eqx.filter_jvp(
            partial(_directional, **self._config()), inputs, directions
        )
        tangent = jax.tree.map(
            lambda value, diff: jnp.zeros_like(value) if diff is None else diff,
            primal,
            tangent,
            is_leaf=lambda x: x is None,
        )
        return primal, tangent

    lin, linearized = hijax.linearize_from_jvp
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

    def transpose(self, out_ct, *args):
        """Apply A.T to vector cotangents and the shared pairing pullback to parameters."""
        model, state, rate, direction = eqx.combine(
            args, cast(tuple, self.params["static"])
        )
        if _has_accum((model, state, rate)):
            raise ValueError(
                "linearized demographic actions are linear only in their directions"
            )
        t_model, t_state, t_rate = direction
        cotangent = jax.tree.map(hijax.instantiate_zeros, out_ct, is_leaf=_is_ad_leaf)
        if _has_accum(t_state):
            _accumulate(
                t_state,
                _bind_action(
                    model,
                    cotangent,
                    rate,
                    segregating=self.params["segregating"],
                    transpose=not self.params["orientation"],
                ),
            )
        if _has_accum((t_model, t_rate)):
            gradients = _parameter_pullback(
                model, state, rate, cotangent, (t_model, t_rate), **self._config()
            )
            _accumulate((t_model, t_rate), gradients)

    def batch_dim_rule(self, axis_data, dims):
        """Support parameter batching and tangent-only batching."""
        dim = 0 if any(d is not None for d in jax.tree.leaves(dims)) else None
        return dim, dim


def _bind_action(model, state, rate, *cached, segregating, transpose):
    """Bind an oriented action while retaining array leaves as dynamic inputs."""
    dynamic, static = eqx.partition((model, state, rate, *cached), eqx.is_array)
    avals = jax.tree.map(jax.typeof, dynamic)
    return _DemographicAction(
        avals, static, segregating=segregating, transpose=transpose, cached=bool(cached)
    )(*dynamic)


def _bind_linearized(model, state, rate, direction, *, segregating, transpose):
    """Bind active directions separately so their transpose uses the shared pairing."""
    dynamic, static = eqx.partition((model, state, rate, direction), eqx.is_array)
    avals = jax.tree.map(jax.typeof, dynamic)
    return _DemographicLinearizedAction(
        avals, static, segregating=segregating, transpose=transpose
    )(*dynamic)


def _demographic_advance(model, distribution, *, segregating, singleton_rate, throw):
    """Return forward diagnostics with the shared mathematical derivative attached."""
    stopped_model, stopped_distribution, stopped_rate = _stop_arrays(
        (model, distribution, singleton_rate)
    )
    result = stopped_model._advance(
        stopped_distribution,
        segregating=segregating,
        singleton_rate=stopped_rate,
        throw=throw,
    )
    one = jnp.asarray(1.0, dtype=jnp.float64)
    mass, _ = _bind_action(
        model,
        (distribution, one),
        singleton_rate,
        (result.distribution, one),
        segregating=segregating,
        transpose=True,
    )
    return result._replace(distribution=mass)


def _demographic_expect(model, values, *, segregating, singleton_rate):
    """Return backward values and affine offset with the common derivative contract."""
    state = (values, jnp.asarray(0.0, dtype=jnp.float64))
    stopped_model, stopped_state, stopped_rate = _stop_arrays(
        (model, state, singleton_rate)
    )
    result = _evaluate(
        stopped_model,
        stopped_state,
        stopped_rate,
        segregating=segregating,
        transpose=False,
    )
    return _bind_action(
        model, state, singleton_rate, result, segregating=segregating, transpose=False
    )
