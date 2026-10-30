"""Operator-level power actions with filtered tangent and transpose rules.

Methods supply ``_power_action`` and ``_power_jvp`` in working coordinates.
Preparation, reconstruction, and diagnostics remain outside this boundary.
Sequential uses an exact coupled recurrence. Krylov methods approximate the
Frechet action and its operator pullback without differentiating Arnoldi bases.
"""

# pattern: Functional Core
import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.core
import jax.interpreters.ad as ad
import jax.numpy as jnp
import lineax as lx


def _is_none(value):
    return value is None


def _is_undefined(value):
    return isinstance(value, ad.UndefinedPrimal)


def _materialise(primal, tangent):
    return jnp.zeros_like(primal) if tangent is None else tangent


def _power_impl(operator, initial, steps, method):
    """Execute the method without differentiating its numerical implementation."""
    return method._power_action(operator, initial, steps)


@eqxi.filter_primitive_def
def _power_abstract(operator, initial, steps, method):
    operator, shape = jax.tree.map(
        lambda x: (
            jax.ShapeDtypeStruct(x.shape, x.dtype)
            if isinstance(x, jax.core.ShapedArray)
            else x
        ),
        (operator, initial),
    )
    result = eqx.filter_eval_shape(method._power_action, operator, shape, steps)
    return jax.tree.map(lambda x: jax.core.ShapedArray(x.shape, x.dtype), result)


@eqxi.filter_primitive_jvp
def _power_jvp(primals, tangents):
    operator, initial, steps, method = primals
    t_operator, t_initial, t_steps, t_method = tangents
    assert all(
        x is None for x in jax.tree.leaves((t_steps, t_method), is_leaf=_is_none)
    )
    if any(x is not None for x in jax.tree.leaves(t_operator, is_leaf=_is_none)):
        tangent_operator = lx.TangentLinearOperator(operator, t_operator)
    else:
        tangent_operator = None
    tangent_initial = jax.tree.map(_materialise, initial, t_initial, is_leaf=_is_none)
    return method._power_jvp(
        operator, tangent_operator, initial, tangent_initial, steps
    )


@eqxi.filter_primitive_transpose(materialise_zeros=True)
def _power_transpose(inputs, cotangent):
    operator, initial, steps, method = inputs
    # Operator cotangents arise by transposing the operator-tangent action in
    # the JVP. This rule handles the primitive's linear initial-state argument.
    if any(
        isinstance(x, ad.UndefinedPrimal)
        for x in jax.tree.leaves(operator, is_leaf=_is_undefined)
    ):
        raise ValueError("power actions are linear only in the initial state")
    result = _power_action(method, operator.T, cotangent, steps)
    initial_ct = jax.tree.map(
        lambda x, ct: ct if isinstance(x, ad.UndefinedPrimal) else None,
        initial,
        result,
        is_leaf=_is_undefined,
    )
    return (
        jax.tree.map(lambda _: None, operator),
        initial_ct,
        None,
        jax.tree.map(lambda _: None, method),
    )


_power_p = eqxi.create_vprim(
    "drifft_power_action",
    eqxi.filter_primitive_def(_power_impl),
    _power_abstract,
    _power_jvp,
    _power_transpose,
)
eqxi.register_impl_finalisation(_power_p)


def _power_action(method, operator, initial, steps):
    """Apply a power with method-owned JVPs and an operator-level transpose."""
    method = eqxi.nondifferentiable(method, name="power-action method configuration")
    return eqxi.filter_primitive_bind(_power_p, operator, initial, steps, method)


@eqx.filter_custom_jvp
def _cached_power_action(method, operator, initial, steps, value):
    """Attach the mathematical power rule to an already computed primal value."""
    return value


@_cached_power_action.def_jvp
def _cached_power_jvp(primals, tangents):
    method, operator, initial, steps, value = primals
    t_method, t_operator, t_initial, t_steps, _ = tangents
    if any(
        x is not None for x in jax.tree.leaves((t_method, t_steps), is_leaf=_is_none)
    ):
        raise TypeError(
            "power-action method configuration and exponent are nondifferentiable"
        )
    tangent_operator = None
    if any(x is not None for x in jax.tree.leaves(t_operator, is_leaf=_is_none)):
        tangent_operator = lx.TangentLinearOperator(operator, t_operator)
    tangent_initial = jax.tree.map(_materialise, initial, t_initial, is_leaf=_is_none)
    if tangent_operator is None:
        tangent = _power_action(method, operator, tangent_initial, steps)
    else:
        tangent = _frechet_action(
            method, operator, initial, t_operator, tangent_initial, steps
        )
    return value, tangent


def _frechet_impl(operator, initial, t_operator, t_initial, steps, method):
    result = method._frechet_action(
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


@eqxi.filter_primitive_def
def _frechet_abstract(operator, initial, t_operator, t_initial, steps, method):
    return initial


@eqxi.filter_primitive_jvp
def _frechet_jvp(primals, tangents):
    raise NotImplementedError(
        "Higher-order Krylov sensitivities are not implemented; use Sequential"
    )


@eqxi.filter_primitive_transpose(materialise_zeros=True)
def _frechet_transpose(inputs, cotangent):
    operator, initial, t_operator, t_initial, steps, method = inputs
    # This transpose is a first-order mathematical adjoint. Do not accidentally
    # differentiate the numerical construction of its Krylov factors.
    operator, initial, cotangent = eqxi.nondifferentiable(
        (operator, initial, cotangent),
        msg="Higher-order Krylov sensitivities are not implemented; use Sequential",
    )
    if any(
        _is_undefined(x) for x in jax.tree.leaves(t_operator, is_leaf=_is_undefined)
    ):
        template = jax.tree.map(
            lambda t: (
                jnp.zeros(t.aval.shape, t.aval.dtype) if _is_undefined(t) else None
            ),
            t_operator,
            is_leaf=_is_undefined,
        )
        operator_ct, error, converged = method._frechet_transpose(
            operator, template, initial, cotangent, steps
        )
        operator_ct = eqx.error_if(
            operator_ct,
            ~converged,
            "Krylov operator pullback did not converge; increase dimension or use Sequential",
        )
    else:
        # A known operator direction has no requested cotangent. In particular,
        # an all-None cotangent tree cannot carry an error_if runtime check.
        operator_ct = jax.tree.map(lambda _: None, t_operator)
    if any(_is_undefined(x) for x in jax.tree.leaves(t_initial, is_leaf=_is_undefined)):
        state_ct = _power_action(method, operator.T, cotangent, steps)
        state_ct = jax.tree.map(
            lambda x, ct: ct if _is_undefined(x) else None,
            t_initial,
            state_ct,
            is_leaf=_is_undefined,
        )
    else:
        state_ct = jax.tree.map(lambda _: None, t_initial)
    return (
        jax.tree.map(lambda _: None, operator),
        jax.tree.map(lambda _: None, initial),
        operator_ct,
        state_ct,
        None,
        jax.tree.map(lambda _: None, method),
    )


_frechet_p = eqxi.create_vprim(
    "drifft_frechet_action",
    eqxi.filter_primitive_def(_frechet_impl),
    _frechet_abstract,
    _frechet_jvp,
    _frechet_transpose,
)
eqxi.register_impl_finalisation(_frechet_p)


def _frechet_action(method, operator, initial, t_operator, t_initial, steps):
    """Apply the linearized power as a primitive with an explicit adjoint."""
    return eqxi.filter_primitive_bind(
        _frechet_p, operator, initial, t_operator, t_initial, steps, method
    )
