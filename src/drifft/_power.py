"""Operator-level power actions with filtered tangent and transpose rules.

Methods supply ``_power_action`` and ``_power_jvp`` in working coordinates.
Preparation, reconstruction, and diagnostics remain outside this boundary.
Sequential implements the contract first; Krylov methods retain their existing
AD paths until their tangent-action transposes are implemented and validated.
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
