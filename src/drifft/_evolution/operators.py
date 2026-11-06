"""Operator parameter derivatives, independent of power algorithms and history."""

# pattern: Functional Core
from functools import singledispatch

import equinox as eqx
import jax
import lineax as lx


@singledispatch
def _numerical_operator(operator):
    """Preserve generic operator structure; specializations may assemble a cache.

    Dispatch only on prepared Lineax operators, never opaque HiType tracers.
    Callers stop numerical inputs before projection; this function itself does
    not discard parameter derivatives from ordinary operator evaluation.
    """
    return operator


def _operator_pushforward(operator, direction):
    """Represent DA[direction] without requiring a dense matrix or Jacobian."""
    return lx.TangentLinearOperator(operator, direction)


def _contract_operator_factors(operator, template, left, right):
    """Apply DA* to fixed factors left @ right.T in active PyTree coordinates.

    The template retains None for inactive leaves. Only the operator is a VJP
    argument; the factors are residuals, not differentiated during contraction.
    """
    selected = jax.tree.map(
        lambda x: x is not None, template, is_leaf=lambda x: x is None
    )
    if not any(jax.tree.leaves(selected)):
        return jax.tree.map(lambda _: None, operator)
    active, fixed = eqx.partition(operator, selected, is_leaf=lambda x: x is None)

    def action(parameters):
        op = eqx.combine(parameters, fixed)
        return jax.lax.map(op.mv, right.T)

    _, pullback = eqx.filter_vjp(action, active)
    return pullback(left.T)[0]
