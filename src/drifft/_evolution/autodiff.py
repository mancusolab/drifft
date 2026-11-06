r"""Differentiate working-coordinate powers through mathematical action rules.

For a represented operator $A(\theta)$, initial state $x$ and fixed nonnegative
integer $k$, the target is $F(\theta,x)=A(\theta)^k x$. With
$E=\mathrm D A(\theta)[\dot\theta]$, its directional derivative is

$$
\dot F=A^k\dot x+
\sum_{j=0}^{k-1} A^{k-1-j} E A^j x,
$$

where the sum is empty for $k=0$. Reverse mode applies $(A^T)^k$ to the output
cotangent for the state contribution and contracts the operator adjoint into
the active parameter coordinates. These identities apply to either orientation
and to operators augmented with a source coordinate.

The HiJAX primitives here define the JAX transformation contracts for those
operations. Private method capabilities in `methods.py` select how to evaluate
these identities, using numerical kernels in `sensitivity.py`. The power JVP
has no solver-family dispatch. Projected sensitivities retain an explicit
mathematical transpose rather than differentiating adaptive basis construction.
A cached primal can supply the value without adding its numerical computation
as a second derivative path.

Operator parameterization is a separate concern: `operators.py` provides
generic action derivatives, while `parameterized.py` supplies Jacobi's
parameter-space representation and contractions. Ordinary operator PyTrees
and opaque HiTypes need separate bindings because their tangent and batching
structures differ; they share the method's checked sensitivity capabilities.
Preparation and reconstruction remain outside the power boundary, so their
parameter dependence is included by the enclosing history chain rule in
`_demographic_action.py`.

Sequential supports higher derivatives where the underlying operations do.
Further differentiation through Krylov sensitivity/pullback primitives is
explicitly unsupported; this does not preclude supported mixed state/parameter
derivatives. Primal convergence does not certify sensitivity convergence.
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


class _PowerMethod[State](Protocol):
    """Private evaluation capabilities consumed by the shared power rule.

    Built-in methods choose numerical evaluation without changing the power
    derivative. This protocol is separate from the public epoch-method contract:
    custom methods may differentiate their own implementation of `solve`.
    """

    def _power_action(
        self, operator: lx.AbstractLinearOperator, initial: State, steps: int
    ) -> State:
        """Evaluate a checked working-coordinate power."""
        ...

    def _linearize_power(
        self, operator, initial: State, operator_direction, tangent: State, steps: int
    ) -> tuple[State | None, State]:
        """Return an optional reusable primal followed by the joint direction.

        The first result is None when evaluation should retain the existing
        primal binding; a fused evaluator may supply its computed primal.
        `operator_direction` is a filtered operator PyTree with inactive leaves
        set to None. The second result is D(A**k x)[operator_direction, tangent]
        and must preserve the method's supported AD contracts.
        """
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


class _PowerAction(hijax.HiPrim):
    r"""Bind $A^k x$ with operator-PyTree and initial-state derivatives.

    The operator's active array leaves describe its differentiable coordinates;
    $k$ and numerical-method configuration are fixed. The JVP combines the
    initial-state action $A^k\dot x$ with the Frechet power action generated by
    $\dot A$. A missing operator direction reduces to an ordinary power action,
    without an operator-sensitivity calculation.

    Sequential evaluates a coupled recurrence; Krylov uses `_FrechetAction`
    so adaptive basis construction does not become the derivative target.
    An optional cached value supplies the primal only: its tangent is ignored.
    Reverse rules are derived from the JVP, while standalone transposition is
    allowed only in the initial state of an uncached power. This boundary lets
    numerical evaluation change without redefining the power derivative.
    """

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
        """Apply the power derivative using the selected method family's evaluation."""
        operator, initial, steps, method, *value = eqx.combine(
            primals, cast(tuple, self.params["static"])
        )
        t_operator, t_initial, t_steps, t_method, *_ = tangents
        if _has_tangent((t_steps, t_method)):
            raise TypeError(
                "power-action method configuration and exponent are nondifferentiable"
            )
        operator_direction = _operator_tangent(t_operator)
        tangent_initial = jax.tree.map(
            hijax.instantiate_zeros, t_initial, is_leaf=_is_ad_leaf
        )
        if not _has_tangent(t_operator):
            primal = self(*primals)
            tangent = _power_action(method, operator, tangent_initial, steps)
        else:
            value, tangent = method._linearize_power(
                operator, initial, operator_direction, tangent_initial, steps
            )
            # A fused evaluator may also produce the primal. Rebind that value
            # to preserve higher derivatives without repeating the computation.
            primal = (
                self(*primals)
                if value is None
                else _bind_power(operator, initial, steps, method, value)
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


class _FrechetAction(hijax.HiPrim):
    r"""Represent the linearized power in ordinary operator coordinates.

    For fixed $A,x,k$, the mathematical action is
    $A^k\dot x+\sum_{j=0}^{k-1}A^{k-1-j}\dot A A^j x$.
    It is linear in the operator direction and state direction, even though
    the Krylov projection used to approximate it can depend on those directions.
    The explicit transpose therefore uses mathematical adjoint factors and the
    opposite state action, rather than differentiating that numerical projection.

    Operator cotangents are contracted into active PyTree leaves by the generic
    operator adapter. State cotangents use $(A^T)^k$ independently, and failed
    sensitivity checks raise. Further differentiation through this sensitivity
    primitive is rejected explicitly; direct transposition is restricted to its
    tangent arguments. This separates first-order mathematical AD from the
    implementation of its approximate numerical evaluation.
    """

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
        return method._sensitivity_action(
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
            left, right = method._sensitivity_adjoint_factors(
                operator, initial, cotangent, steps
            )
            operator_ct = _contract_operator_factors(operator, template, left, right)
            _accumulate(t_operator, operator_ct)
        if _has_accum(t_initial):
            _accumulate(t_initial, _power_action(method, operator.T, cotangent, steps))

    def batch_dim_rule(self, axis_data, dims):
        """Map sensitivity outputs only when an input is mapped."""
        dim = 0 if any(x is not None for x in jax.tree.leaves(dims)) else None
        return jax.tree.map(lambda _: dim, self.out_aval)


def _frechet_action(method, operator, initial, t_operator, t_initial, steps):
    """Bind the linearized power with an explicit adjoint."""
    dynamic, static = eqx.partition(
        (operator, initial, t_operator, t_initial, steps, method), eqx.is_array
    )
    avals = jax.tree.map(jax.typeof, dynamic)
    return _FrechetAction(avals, static)(*dynamic)


class _Power(hijax.HiPrim):
    r"""Bind $A(\theta)^k x$ with a parameter-space operator tangent.

    The primal operator is an opaque value containing parameters, fixed geometry
    and a matrix cache, but its tangent is a parameter vector rather than a
    matrix. The power product rule is the same as `_PowerAction`; only the
    operator-coordinate representation and its JAX binding differ.

    At a cached binding, visible parameter leaves preserve which process
    coordinates are active before flattening. They own the parameter tangent;
    adding the opaque value's aliased tangent would count it twice. The opaque
    primal remains live so supported mixed state/parameter derivatives survive.
    Cached outputs contribute no additional derivative path.

    Parameter directions invoke `_Sensitivity`; state-only directions invoke a
    power with the same operator. The standalone transpose applies the opposite
    orientation only for an uncached action linear in its initial state.
    """

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
    r"""Linearize a parameterized power without differentiating its matrix cache.

    Given $\dot\theta$, assembly supplies $E=\mathrm D A(\theta)[\dot\theta]$.
    The returned state direction approximates
    $A^k\dot x+\sum_{j=0}^{k-1}A^{k-1-j}EA^j x$ with the shared checked Krylov
    kernel. Fixed geometry is not a derivative coordinate.

    Transposition returns parameter coordinates through `_Pullback` and state
    coordinates through the opposite power action. The active-process metadata
    limits derivative assembly; it records structural participation, not whether
    a direction happens to be numerically zero. Only tangent arguments admit
    standalone transposition, and further AD through this primitive is unsupported.
    A separate binding preserves these contracts without forcing compact
    parameter tangents into an ordinary operator PyTree.
    """

    def __init__(self, avals, method, steps, *, active_process=None):
        self.in_avals = avals
        self.out_aval = avals[1]
        self.params = dict(method=method, steps=steps, active_process=active_process)
        super().__init__()

    def expand(self, *args):
        operator, initial, dp, dx = args
        operator = _unpack(operator)
        return cast(Any, self.params["method"])._sensitivity_action(
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
    r"""Return parameter cotangents for a power from fixed primal and adjoint seeds.

    For $y=A(\theta)^k x$ and output cotangent $g$, the result approximates
    $\nabla_\theta[g^T A(\theta)^k x]$ with $x,g$ held fixed. Krylov spaces
    produce operator-adjoint factors $L,R$; the representation adapter contracts
    $LR^T$ directly into process/source coordinates. No dense matrix cotangent
    or reverse tape for matrix assembly is constructed.

    Only active process directions are evaluated, and the factor calculation
    must pass its convergence check. The initial-state cotangent is computed
    separately by the calling sensitivity rule. Further JVP/VJP differentiation
    and standalone transposition of this primitive are explicitly unsupported;
    its three-input contract must not inherit a tangent-action transpose rule.
    """

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
        left, right = cast(Any, self.params["method"])._sensitivity_adjoint_factors(
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
