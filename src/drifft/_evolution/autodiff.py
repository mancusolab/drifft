"""HiJAX power actions with mathematical sensitivities and explicit adjoints.

Preparation, reconstruction, and diagnostics remain outside these primitives.
[`Sequential`][drifft.Sequential] uses an exact coupled recurrence. Krylov methods approximate the
Frechet action and its operator pullback without differentiating Arnoldi bases.
"""

# pattern: Functional Core
from typing import Protocol, cast

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import lineax as lx
from jax.experimental import hijax

from .problem import _KrylovConfig
from .sensitivity import _frechet_operator_pullback, _frechet_power


class _PowerMethod[State](Protocol):
    """Numerical power evaluation consumed by the shared HiJAX rule."""

    def _power_action(
        self, operator: lx.AbstractLinearOperator, initial: State, steps: int
    ) -> State:
        """Evaluate a checked working-coordinate power."""
        ...


class _KrylovMethod(_PowerMethod[jax.Array], _KrylovConfig, Protocol):
    """Power callbacks plus the settings needed by the Krylov derivative kernels."""


def _is_ad_leaf(value):
    """Keep symbolic zeros and cotangent accumulators opaque during tree walks."""
    return isinstance(value, (hijax.Zero, hijax.GradAccum))


def _has_accum(tree):
    """Identify arguments whose cotangents are requested by transposition."""
    return any(
        isinstance(x, hijax.GradAccum)
        for x in jax.tree.leaves(tree, is_leaf=_is_ad_leaf)
    )


def _tangent_or_none(tangent):
    """Translate a symbolic zero into Equinox's inactive-leaf convention."""
    return None if isinstance(tangent, hijax.Zero) else tangent


def _operator_tangent(tangent):
    """Preserve operator structure while marking unperturbed leaves as inactive."""
    return jax.tree.map(_tangent_or_none, tangent, is_leaf=_is_ad_leaf)


def _has_tangent(tree):
    """Check for active directions without materializing symbolic zero arrays."""
    return any(
        not isinstance(x, hijax.Zero)
        for x in jax.tree.leaves(tree, is_leaf=_is_ad_leaf)
    )


def _accumulate(accums, values):
    """Add cotangents only to leaves participating in the current transpose."""

    def add(accum, value):
        """Leave known arguments untouched while accumulating active leaves."""
        if isinstance(accum, hijax.GradAccum):
            accum.accum(value)

    jax.tree.map(add, accums, values, is_leaf=lambda x: x is None or _is_ad_leaf(x))


def _sequential_power_direction(operator, perturbation, initial, tangent, steps):
    """Evaluate D(A**g x)[dA, dx] by the coupled mathematical recurrence."""

    def step(_, carry):
        state, direction = carry
        return operator.mv(state), jax.tree.map(
            jnp.add, operator.mv(direction), perturbation.mv(state)
        )

    return jax.lax.fori_loop(0, steps, step, (initial, tangent))[1]


class _PowerAction(hijax.HiPrim):
    """Power evaluation or cached value with the same mathematical derivative."""

    def __init__(self, avals, static, *, cached):
        """Keep array leaves dynamic and filtered Python configuration static."""
        self.in_avals = avals
        self.out_aval = avals[1]
        self.params = dict(static=static, cached=cached)
        super().__init__()

    def expand(self, *args):
        """Compute a power or retain the primal already computed with status."""
        operator, initial, steps, method, *value = eqx.combine(
            args, cast(tuple, self.params["static"])
        )
        if self.params["cached"]:
            return value[0]
        return method._power_action(operator, initial, steps)

    def jvp(self, primals, tangents):
        """Differentiate the mathematical power, independently of solver implementation."""
        operator, initial, steps, method, *value = eqx.combine(
            primals, cast(tuple, self.params["static"])
        )
        t_operator, t_initial, t_steps, t_method, *_ = tangents
        if _has_tangent((t_steps, t_method)):
            raise TypeError(
                "power-action method configuration and exponent are nondifferentiable"
            )
        operator_direction = _operator_tangent(t_operator)
        tangent_operator = (
            lx.TangentLinearOperator(operator, operator_direction)
            if _has_tangent(t_operator)
            else None
        )
        tangent_initial = jax.tree.map(
            hijax.instantiate_zeros, t_initial, is_leaf=_is_ad_leaf
        )
        primal = self(*primals)
        if tangent_operator is None:
            tangent = _power_action(method, operator, tangent_initial, steps)
        elif isinstance(method, _KrylovConfig):
            # Numerical evaluation of the same power derivative by a projected
            # block action; its adaptive basis is never the differentiation target.
            tangent = _frechet_action(
                method, operator, initial, operator_direction, tangent_initial, steps
            )
        else:
            tangent = _sequential_power_direction(
                operator, tangent_operator, initial, tangent_initial, steps
            )
        return primal, tangent

    lin, linearized = hijax.linearize_from_jvp
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

    def transpose(self, out_ct, *args):
        """Transpose only the linear initial-state argument of an uncached power."""
        operator, initial, steps, method, *_ = eqx.combine(
            args, cast(tuple, self.params["static"])
        )
        if self.params["cached"] or _has_accum((operator, steps, method)):
            raise ValueError("power actions are linear only in the initial state")
        if _has_accum(initial):
            cotangent = jax.tree.map(
                hijax.instantiate_zeros, out_ct, is_leaf=_is_ad_leaf
            )
            _accumulate(initial, _power_action(method, operator.T, cotangent, steps))

    def batch_dim_rule(self, axis_data, dims):
        """Retain unmapped primal outputs during tangent-only batching."""
        dim = 0 if any(x is not None for x in jax.tree.leaves(dims)) else None
        return jax.tree.map(lambda _: dim, self.out_aval)


def _bind_power(operator, initial, steps, method, *value):
    """Bind array leaves dynamically while preserving static operator structure."""
    dynamic, static = eqx.partition(
        (operator, initial, steps, method, *value), eqx.is_array
    )
    avals = jax.tree.map(jax.typeof, dynamic)
    return _PowerAction(avals, static, cached=bool(value))(*dynamic)


def _power_action[State](
    method: _PowerMethod[State], operator, initial: State, steps: int
) -> State:
    """Apply a power with a shared mathematical JVP and operator-level transpose."""
    return _bind_power(operator, initial, steps, method)


def _cached_power_action(method: _KrylovMethod, operator, initial, steps, value):
    """Attach the mathematical power rule to an already computed primal value."""
    return _bind_power(operator, initial, steps, method, value)


class _FrechetAction(hijax.HiPrim):
    """First-order Krylov sensitivity with an explicit low-rank pullback."""

    def __init__(self, avals, static):
        """Declare the state result and preserve filtered operator structure."""
        self.in_avals = avals
        self.out_aval = avals[1]
        self.params = dict(static=static)
        super().__init__()

    def expand(self, *args):
        """Evaluate the coupled sensitivity and enforce tangent convergence."""
        operator, initial, t_operator, t_initial, steps, method = eqx.combine(
            args, cast(tuple, self.params["static"])
        )
        result = _frechet_power(
            method,
            operator,
            lx.TangentLinearOperator(operator, t_operator),
            initial,
            t_initial,
            steps,
        )
        return eqx.error_if(
            result.distribution,
            ~result.converged,
            "Krylov sensitivity action did not converge; increase dimension or use Sequential",
        )

    def jvp(self, primals, tangents):
        """Reject unsupported higher-order Krylov sensitivities explicitly."""
        raise NotImplementedError(
            "Higher-order Krylov sensitivities are not implemented; use Sequential"
        )

    lin, linearized = hijax.linearize_from_jvp
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

    def transpose(self, out_ct, *args):
        """Accumulate operator and state directions using mathematical adjoints."""
        operator, initial, t_operator, t_initial, steps, method = eqx.combine(
            args, cast(tuple, self.params["static"])
        )
        if _has_accum((operator, initial, steps, method)):
            raise ValueError("Frechet actions are linear only in tangent arguments")
        cotangent = jax.tree.map(hijax.instantiate_zeros, out_ct, is_leaf=_is_ad_leaf)
        operator, initial, cotangent = eqxi.nondifferentiable(
            (operator, initial, cotangent),
            msg="Higher-order Krylov sensitivities are not implemented; use Sequential",
        )
        if _has_accum(t_operator):
            # The template records which operator leaves need a pullback. Known
            # leaves stay None so fixed geometry does not acquire cotangents or
            # trigger an unnecessary operator-sensitivity solve.
            template = jax.tree.map(
                lambda t: (
                    jnp.zeros(t.aval.shape, t.aval.dtype)
                    if isinstance(t, hijax.GradAccum)
                    else None
                ),
                t_operator,
                is_leaf=_is_ad_leaf,
            )
            operator_ct, _, converged = _frechet_operator_pullback(
                method, operator, template, initial, cotangent, steps
            )
            operator_ct = eqx.error_if(
                operator_ct,
                ~converged,
                "Krylov operator pullback did not converge; increase dimension or use Sequential",
            )
            _accumulate(t_operator, operator_ct)
        if _has_accum(t_initial):
            _accumulate(t_initial, _power_action(method, operator.T, cotangent, steps))

    def batch_dim_rule(self, axis_data, dims):
        """Map sensitivity outputs only when an input is mapped."""
        dim = 0 if any(x is not None for x in jax.tree.leaves(dims)) else None
        return jax.tree.map(lambda _: dim, self.out_aval)


def _frechet_action(
    method: _KrylovConfig, operator, initial, t_operator, t_initial, steps
):
    """Bind the linearized power with an explicit adjoint."""
    dynamic, static = eqx.partition(
        (operator, initial, t_operator, t_initial, steps, method), eqx.is_array
    )
    avals = jax.tree.map(jax.typeof, dynamic)
    return _FrechetAction(avals, static)(*dynamic)
