"""Batch HiJAX actions together with their mathematical derivative rules."""

from functools import singledispatch

import equinox.internal as eqxi
import jax
from jax._src.interpreters import batching
from jax.experimental import hijax


@singledispatch
def tangent_axis(dim):
    """Return the mapping of the tangent representation for a primal mapping."""
    return dim


class BatchablePrimitive(hijax.HiPrim):
    """Apply an action's declared output axes across its primal and AD rules."""

    def batch_dim_rule(self, axis_data, dims):
        """Return the mapped axis of each result."""
        raise NotImplementedError

    def batch(self, axis_data, args, dims):
        """Bind a mapped action while preserving its custom derivatives."""
        outputs = self.batch_dim_rule(axis_data, dims)
        mapped = _MappedAction(self, axis_data, dims, outputs)
        return mapped(*args), outputs


class _MappedAction(BatchablePrimitive):
    """A single vmap layer around a prepared primitive."""

    def __init__(self, action, axis, inputs, outputs):
        """Insert mapped dimensions into the declared array and operator types."""
        self.in_avals = hijax.unmap_avals(axis, action.in_avals, inputs)
        self.out_aval = hijax.unmap_avals(axis, action.out_aval, outputs)
        # HiPrim uses params for equality and hashing in JAX's compilation cache.
        self.params = dict(action=action, axis=axis, inputs=inputs, outputs=outputs)
        self.action, self.axis = action, axis
        self.inputs, self.outputs = inputs, outputs
        self.tangent_inputs = jax.tree.map(tangent_axis, inputs)
        self.tangent_outputs = jax.tree.map(tangent_axis, outputs)
        self.skip_linearization_on_zero_tangents = (
            action.skip_linearization_on_zero_tangents
        )
        super().__init__()

    def expand(self, *args):
        """Map the underlying numerical kernel."""
        return hijax.vmap_rule(
            self.axis, self.action.expand, self.inputs, self.outputs
        )(*args)

    def jvp(self, primals, tangents):
        """Map primal and tangent calls with matching axes, including symbolic zeros."""
        return hijax.vmap_rule(
            self.axis,
            self.action.jvp,
            (self.inputs, self.tangent_inputs),
            (self.outputs, self.tangent_outputs),
        )(primals, tangents)

    lin, linearized = hijax.linearize_from_jvp

    def vjp_fwd(self, nzs, *args):
        """Retain the original pullback residuals and their inferred batch axes."""

        def rule(*primals):
            value, residual, *flags = self.action.vjp_fwd(nzs, *primals)
            return value, residual, eqxi.Static(flags)

        (value, residual, flags), (_, dims, _) = hijax.vmap_rule(
            self.axis, rule, self.inputs, (self.outputs, batching.infer, None)
        )(*args)
        return value, (residual, eqxi.Static(dims)), *flags.value

    def vjp_bwd_retval(self, residual, cotangent):
        """Accumulate shared input cotangents and preserve mapped input axes."""
        values, dims = residual
        axes = jax.tree.map(
            lambda d: batching.sum_axis if d is None else d,
            self.tangent_inputs,
            is_leaf=lambda x: x is None,
        )
        return hijax.vmap_rule(
            self.axis,
            self.action.vjp_bwd_retval,
            (dims.value, self.tangent_outputs),
            axes,
            sum_match=True,
        )(values, cotangent)

    def batch_dim_rule(self, axis_data, dims):
        """Express nested input and output mappings in their enclosing coordinates."""

        def shift(d, inner, sign):
            return (
                d
                if d is None or inner is None
                else (d - (inner < d) if sign < 0 else d + (inner < d))
            )

        inner = jax.tree.map(
            lambda d, i: shift(d, i, -1), dims, self.inputs, is_leaf=lambda x: x is None
        )
        outputs = self.action.batch_dim_rule(axis_data, inner)
        return jax.tree.map(
            lambda d, i: shift(d, i, 1),
            outputs,
            self.outputs,
            is_leaf=lambda x: x is None,
        )
