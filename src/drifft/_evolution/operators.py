r"""Differentiate an operator's parameterization without choosing a power algorithm.

A linear action $x\mapsto A(\theta)x$ is linear in the state, but can be nonlinear
in its parameters. Power differentiation therefore needs two distinct operations:
the pushforward $E=\mathrm D A(\theta)[\dot\theta]$, and the pullback from an
operator cotangent to the coordinates used to represent $\theta$.

When the operator cotangent is supplied as factors $G=LR^T$, its parameter
pullback is the derivative of a scalar pairing:

$$
\bar\theta
=\nabla_\theta\langle LR^T,A(\theta)\rangle_F
=\nabla_\theta\langle L,A(\theta)R\rangle_F,
$$

holding $L$ and $R$ fixed during this contraction. Here $\langle\cdot,\cdot\rangle_F$
is the real Frobenius inner product. The generic implementation uses Lineax
action JVPs and filtered VJPs, so it does not require materializing $A$ or its
Jacobian. Cotangents follow the active operator PyTree leaves; they are not
necessarily small if those leaves include a dense matrix.

This module also selects the numerical representation of a prepared operator,
preserving generic matrix-free or structured operators by default. Jacobi's
specialization in `parameterized.py` can expose a matrix cache and contract
adjoints directly into process/source coordinates. Power rules and cached
primitive binding belong to `autodiff.py`; neither history composition nor
Krylov projection construction belongs to this parameterization boundary.
"""

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
