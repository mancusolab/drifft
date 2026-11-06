"""HiJAX power actions with mathematical sensitivities and explicit adjoints.

Preparation, reconstruction, and diagnostics remain outside these primitives.
[`Sequential`][drifft.Sequential] uses an exact coupled recurrence. Krylov methods
approximate the Frechet action and its operator pullback without differentiating
Arnoldi bases.

Ordinary operator PyTrees and parameter-space HiTypes have distinct bindings,
sharing the checked sensitivity kernels. Operator-coordinate pushforwards and
factor contractions live in operators.py and parameterized.py; this module owns
power differentiation and selects cached bindings from prepared operator types.
"""

# pattern: Functional Core
from functools import singledispatch
from typing import Any, Protocol, cast

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import lineax as lx
from jax.experimental import hijax

from .operators import _contract_operator_factors, _operator_pushforward
from .parameterized import (
    _contract_parameter_factors,
    _JacobiOperator,
    _pack,
    _parameter_direction,
    _parameter_pushforward,
    _transpose,
    _unpack,
)
from .problem import _KrylovConfig
from .sensitivity import _frechet_adjoint_factors, _frechet_power

# Autodiff design: differentiate the represented mathematical operation, not the
# algorithm used to approximate it. At the history level, advance applies H.T to
# mass (plus accumulated sources), while expect applies H to terminal values and
# returns the source reward. Their vector pullbacks use the opposite orientation.
# History composition includes parameter-dependent encoding, reconstruction,
# boundaries and sources; differentiating a reduced matrix alone is insufficient.
#
# This module supplies the epoch power rule for y = A(theta)**k @ x. Its JVP is
# A**k @ dx + D(A**k)[dA] @ x; its VJP combines (A.T)**k @ output_ct with a
# contraction of the operator adjoint into theta. The derivative identity is
# shared. Sequential evaluates a coupled recurrence; Krylov approximates the
# Frechet action/adjoint without differentiating adaptive projection bases.
# Cached primal values avoid repeating numerical work, but their computation
# does not supply an additional tangent. Numerical diagnostics are not AD targets.
#
# Operator parameterization owns dA and the adjoint contraction, not the power
# rule. Generic Lineax operators use action JVPs/VJPs and remain matrix-free when
# appropriate. The Jacobi parameter-space representation instead streams active
# assembly directions and contracts source coordinates directly, avoiding an
# assembly reverse tape or a dense matrix cotangent. Fixed geometry and matrix
# caches remain primal numerical state, not independent derivative coordinates.
#
# We retain separate PyTree and HiType bindings because their tangent structures
# and batching contracts differ, while sharing numerical sensitivity kernels and
# convergence checks. Forcing one binding would obscure those contracts; separate
# backend-specific power mathematics would duplicate them. Representation details
# belong in operators.py/parameterized.py, numerical kernels in sensitivity.py,
# and history rules in _demographic_action.py. Preserve Sequential higher AD and
# supported mixed derivatives; further Krylov operator sensitivities are explicitly
# unsupported rather than falling through to differentiation of the solver.


class _PowerMethod[State](Protocol):
    """Numerical power evaluation consumed by the shared HiJAX rule."""

    def _power_action(
        self, operator: lx.AbstractLinearOperator, initial: State, steps: int
    ) -> State:
        """Evaluate a checked working-coordinate power."""
        ...


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


def _sequential_power_pair(operator, perturbation, initial, tangent, steps):
    """Evaluate A**g x and its joint derivative in one coupled recurrence."""

    def step(_, carry):
        state, direction = carry
        return operator.mv(state), jax.tree.map(
            jnp.add, operator.mv(direction), perturbation.mv(state)
        )

    return jax.lax.fori_loop(0, steps, step, (initial, tangent))


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
            _operator_pushforward(operator, operator_direction)
            if _has_tangent(t_operator)
            else None
        )
        tangent_initial = jax.tree.map(
            hijax.instantiate_zeros, t_initial, is_leaf=_is_ad_leaf
        )
        if tangent_operator is None:
            primal = self(*primals)
            tangent = _power_action(method, operator, tangent_initial, steps)
        elif isinstance(method, _KrylovConfig):
            primal = self(*primals)
            # Numerical evaluation of the same power derivative by a projected
            # block action; its adaptive basis is never the differentiation target.
            tangent = _frechet_action(
                method, operator, initial, operator_direction, tangent_initial, steps
            )
        else:
            value, tangent = _sequential_power_pair(
                operator, tangent_operator, initial, tangent_initial, steps
            )
            # Preserve the primitive boundary for higher derivatives while
            # reusing the primal already produced by the coupled recurrence.
            primal = _bind_power(operator, initial, steps, method, value)
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


@singledispatch
def _cached_power_action(operator, method, initial, steps, value, numerical_operator):
    """Bind a checked primal using the prepared operator's derivative coordinates.

    The generic path retains the differentiable operator PyTree. Specializations
    can also reuse numerical_operator, the stopped representation used by the
    primal projection, without rebuilding it or differentiating its cache.
    """
    return _bind_power(operator, initial, steps, method, value)


@_cached_power_action.register(_JacobiOperator)
def _cached_parameterized_power(
    operator, method, initial, steps, value, numerical_operator
):
    """Reuse the projected matrix while returning cotangents in parameter space."""
    return parameterized_power(
        method, operator, initial, steps, value, numerical_operator.as_matrix()
    )


def _checked_frechet_action(method, operator, perturbation, initial, tangent, steps):
    """Evaluate the shared power derivative and enforce its convergence contract."""
    result = _frechet_power(method, operator, perturbation, initial, tangent, steps)
    return eqx.error_if(
        result.distribution,
        ~result.converged,
        "Krylov sensitivity action did not converge; increase dimension or use Sequential",
    )


def _checked_adjoint_factors(method, operator, initial, cotangent, steps):
    """Keep numerical factor evaluation inside the derivative primitive boundary."""
    left, right, _, converged = _frechet_adjoint_factors(
        method, operator, initial, cotangent, steps
    )
    left = eqx.error_if(
        left,
        ~converged,
        "Krylov operator pullback did not converge; increase dimension or use Sequential",
    )
    return left, right


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
        return _checked_frechet_action(
            method,
            operator,
            _operator_pushforward(operator, t_operator),
            initial,
            t_initial,
            steps,
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
            left, right = _checked_adjoint_factors(
                method, operator, initial, cotangent, steps
            )
            operator_ct = _contract_operator_factors(operator, template, left, right)
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


class _Power(hijax.HiPrim):
    """Apply a mathematical power or attach its rule to a cached primal."""

    def __init__(self, avals, method, steps, cached, *, with_parameter_leaves=False):
        self.in_avals = avals
        self.out_aval = avals[1]
        self.params = dict(
            method=method,
            steps=steps,
            cached=cached,
            with_parameter_leaves=with_parameter_leaves,
        )
        super().__init__()

    def expand(self, *args):
        operator, initial, *value = args
        if self.params["with_parameter_leaves"]:
            _, *value = value
        operator = _unpack(operator)
        if self.params["cached"]:
            return value[0]
        return cast(Any, self.params["method"])._power_action(
            lx.MatrixLinearOperator(operator.matrix), initial, self.params["steps"]
        )

    def jvp(self, primals, tangents):
        operator, initial, *_ = primals
        dp, dx, *_ = tangents
        active_process = None
        if self.params["with_parameter_leaves"]:
            # These leaves alias the opaque payload, but retain structural zeros
            # until this boundary. They own the parameter tangent here; using
            # the opaque tangent as well would count that contribution twice.
            leaves, directions = primals[2], tangents[2]
            dp, active_process = _parameter_direction(
                leaves,
                directions,
                jax.typeof(operator).spec.process_size,
                jax.typeof(operator).to_tangent_aval(),
            )
        if isinstance(dp, hijax.Zero):
            tangent = _power(operator, hijax.instantiate_zeros(dx), **self._settings())
        else:
            args = operator, initial, dp, hijax.instantiate_zeros(dx)
            tangent = _Sensitivity(
                jax.tree.map(jax.typeof, args),
                active_process=active_process,
                **self._settings(),
            )(*args)
        return self(*primals), tangent

    def _settings(self):
        return {key: self.params[key] for key in ("method", "steps")}

    lin, linearized = hijax.linearize_from_jvp
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

    def transpose(self, out_ct, *args):
        ct = out_ct
        operator, initial, *value = args
        if isinstance(operator, hijax.GradAccum) or value:
            raise ValueError("power is linear only in the initial state")
        if isinstance(initial, hijax.GradAccum):
            initial.accum(
                _power(
                    _transpose(operator),
                    hijax.instantiate_zeros(ct),
                    **self._settings(),
                )
            )

    def batch_dim_rule(self, axis_data, dims):
        return 0 if any(d is not None for d in jax.tree.leaves(dims)) else None


class _Sensitivity(hijax.HiPrim):
    """Linearized power with direct parameter-space adjoint contraction."""

    def __init__(self, avals, method, steps, *, active_process=None):
        self.in_avals = avals
        self.out_aval = avals[1]
        self.params = dict(method=method, steps=steps, active_process=active_process)
        super().__init__()

    def expand(self, *args):
        operator, initial, dp, dx = args
        operator = _unpack(operator)
        return _checked_frechet_action(
            self.params["method"],
            lx.MatrixLinearOperator(operator.matrix),
            _parameter_pushforward(operator, dp),
            initial,
            dx,
            self.params["steps"],
        )

    def jvp(self, primals, tangents):
        raise NotImplementedError(
            "Higher-order Krylov sensitivities are not implemented; use Sequential"
        )

    lin, linearized = hijax.linearize_from_jvp
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

    def _settings(self):
        return {key: self.params[key] for key in ("method", "steps")}

    def batch_dim_rule(self, axis_data, dims):
        return 0 if any(d is not None for d in jax.tree.leaves(dims)) else None

    def transpose(self, out_ct, *args):
        ct = out_ct
        operator, initial, dp, dx = args
        if isinstance(operator, hijax.GradAccum) or isinstance(
            initial, hijax.GradAccum
        ):
            raise ValueError("sensitivity is linear only in tangent arguments")
        cotangent = hijax.instantiate_zeros(ct)
        if isinstance(dp, hijax.GradAccum):
            args = operator, initial, cotangent
            dp.accum(
                _Pullback(
                    jax.tree.map(jax.typeof, args),
                    active_process=self.params["active_process"],
                    **self._settings(),
                )(*args)
            )
        if isinstance(dx, hijax.GradAccum):
            dx.accum(_power(_transpose(operator), cotangent, **self._settings()))


class _Pullback(hijax.HiPrim):
    """Contract power-adjoint factors; further AD and transposition are unsupported."""

    def __init__(self, avals, method, steps, *, active_process=None):
        self.in_avals = avals
        self.out_aval = avals[0].to_ct_aval()
        self.params = dict(method=method, steps=steps, active_process=active_process)
        super().__init__()

    def jvp(self, primals, tangents):
        raise NotImplementedError(
            "Higher-order Krylov sensitivities are not implemented; use Sequential"
        )

    lin, linearized = hijax.linearize_from_jvp
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

    def transpose(self, out_ct, *args):
        raise NotImplementedError(
            "Higher-order Krylov sensitivities are not implemented; use Sequential"
        )

    def batch_dim_rule(self, axis_data, dims):
        return 0 if any(d is not None for d in jax.tree.leaves(dims)) else None

    def expand(self, *args):
        operator, initial, cotangent = args
        operator = _unpack(operator)
        left, right = _checked_adjoint_factors(
            self.params["method"],
            lx.MatrixLinearOperator(operator.matrix),
            initial,
            cotangent,
            self.params["steps"],
        )
        return _contract_parameter_factors(
            operator, left, right, self.params["active_process"]
        )


def _power(operator, initial, method, steps, *value):
    args = operator, initial, *value
    return _Power(jax.tree.map(jax.typeof, args), method, steps, bool(value))(*args)


def parameterized_power(method, operator, initial, steps, value, matrix):
    """Attach a parameter-space mathematical rule to a stopped numerical cache."""
    leaves = tuple(
        jax.tree.leaves(
            eqx.filter((operator.frequency_map, operator.source), eqx.is_array)
        )
    )
    args = _pack(operator, matrix), initial, leaves, value
    return _Power(
        jax.tree.map(jax.typeof, args), method, steps, True, with_parameter_leaves=True
    )(*args)
