r"""Mathematical history actions and their shared differentiation boundary.

Let $H(\theta)$ be the represented history expectation operator and $c(\theta)$
the final mass accumulated from singleton sources starting from zero mass.
For initial mass $p$ and terminal values $f$, the public actions are

$$
\operatorname{advance}(p)=H^T p+c, \qquad
\operatorname{expect}(f)=(Hf,\;c^T f).
$$

The extra source term makes forward evolution affine. Augmenting the state
with a scalar makes both orientations linear actions of one operator:
$\mathcal A=\begin{pmatrix}H^T&c\\0&1\end{pmatrix}$.
Forward evolution applies $\mathcal A$ to $(p,1)^T$; backward expectation
applies $\mathcal A^T$ to $(f,0)^T$. Vector pullbacks use the opposite
orientation, while parameter pullbacks differentiate the scalar pairing
$y^T\mathcal A(\theta)x$ with the augmented vectors held fixed.

This boundary keeps the history derivative independent of the choice of state
representation. The traversal in actions.py composes epoch powers and population-size
boundaries, including parameter-dependent encoding, reconstruction and sources.
The power rules in `_evolution/autodiff.py` supply epoch sensitivities; this
module retains parameter linearizations from the requested primal orientation
rather than forming $H$ or a history Jacobian. Geometry, numerical settings and
diagnostics are not differentiation targets.

$H$ denotes the numerical representation of the Wright–Fisher history, not
necessarily its exact transition matrix. Krylov action and sensitivity errors
can differ: opposite orientations approximate the mathematical transpose,
rather than exactly transposing a particular adaptive projection.
"""

# pattern: Functional Core
from typing import cast

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.experimental import hijax

from .._evolution.autodiff import (
    _accumulate,
    _has_accum,
    _has_tangent,
    _is_ad_leaf,
    _tangent_or_none,
)
from .._evolution.problem import EvolutionStatus
from .actions import (
    DemographicResult,
    IntervalDiagnostics,
    _advance_history,
    _expect_history,
)


def _stop_arrays(tree):
    """Keep primal diagnostics and adaptive numerical construction outside AD."""
    arrays, static = eqx.partition(tree, eqx.is_array)
    return eqx.combine(jax.tree.map(jax.lax.stop_gradient, arrays), static)


def _evaluate_with_aux(
    model, state, rate, *, segregating, transpose, diagnostics, throw
):
    """Evaluate once and separate nondifferentiable numerical diagnostics."""
    if transpose:
        vector, scalar = state
        result = _advance_history(
            model,
            vector,
            segregating=segregating,
            singleton_rate=rate * scalar,
            throw=throw,
        )
        payload = (
            (
                result.status,
                tuple(
                    (item.error_estimate, item.status) for item in result.diagnostics
                ),
            )
            if diagnostics
            else None
        )
        return (result.distribution, scalar), _stop_arrays(payload)
    return _expect_history(model, state, rate, segregating=segregating), None


def _parameter_vjp(
    model, seed, rate, mask, *, segregating, transpose, diagnostics=False, throw=True
):
    """Linearize parameters during the requested orientation's primal evaluation.

    Contracting this pullback with an output cotangent differentiates the same
    augmented scalar pairing in either orientation. Epoch primitives supply
    mathematical derivatives; adaptive solver construction remains outside AD.
    """
    active, fixed = eqx.partition((model, rate), mask)

    def action(parameters):
        current_model, current_rate = eqx.combine(parameters, fixed)
        return _evaluate_with_aux(
            current_model,
            seed,
            current_rate,
            segregating=segregating,
            transpose=transpose,
            diagnostics=diagnostics,
            throw=throw,
        )

    return eqx.filter_vjp(action, active, has_aux=True)


def _pack(state, auxiliary, diagnostics):
    return (state, auxiliary) if diagnostics else state


def _parameter_mask(nzs):
    return jax.tree.map(
        lambda active: False if active is None else active,
        (nzs[0], nzs[2]),
        is_leaf=lambda x: x is None,
    )


def _check_fixed(model_flags):
    if any(jax.tree.leaves((model_flags.geometry, model_flags.method))):
        raise TypeError(
            "demographic geometry and numerical method configuration are nondifferentiable"
        )


class _DemographicAction(hijax.HiPrim):
    r"""Apply an oriented augmented history action with mathematical AD rules.

    With $\mathcal A=\begin{pmatrix}H^T&c\\0&1\end{pmatrix}$, the primitive
    applies either $\mathcal A x$ or $\mathcal A^T x$. The scalar coordinate
    scales source input in the forward orientation and accumulates source
    reward in the backward orientation. This makes affine histories linear in
    the augmented vector, including arbitrary scalar cotangents in reverse mode.

    JVPs compose the history's epoch and boundary derivatives. Parameter VJPs
    retain the linearization of the requested primal orientation; vector VJPs
    invoke the opposite action. Geometry and method settings are fixed, and
    diagnostic cotangents do not contribute. A cached value preserves the same
    AD boundary without a second primal evaluation. Standalone transposition
    is restricted to the augmented vector of an uncached, diagnostic-free action.

    The primitive targets the represented history, including source, encoding
    and reconstruction dependence. Approximate epoch solves can give different
    forward/backward numerical errors; this rule is not the exact transpose of
    an input-dependent Krylov projection.
    """

    def __init__(
        self, avals, static, *, segregating, transpose, diagnostics, throw, cached
    ):
        """Declare the population output and optional interval diagnostic payload."""
        self.in_avals = avals
        model = eqx.combine(avals[0], static[0])
        size = (
            model.demography.final_pop_size
            if transpose
            else model.demography.initial_pop_size
        )
        state_aval = (
            ShapedArray((2 * size + (-1 if segregating else 1),), jnp.float64),
            ShapedArray((), jnp.float64),
        )
        status_aval = jax.tree.map(jax.typeof, EvolutionStatus.successful)
        auxiliary_aval = (
            status_aval,
            tuple(
                (ShapedArray((), jnp.float64), status_aval)
                for _ in model.demography.generations
            ),
        )
        self.out_aval = _pack(state_aval, auxiliary_aval, diagnostics)
        self.params = dict(
            static=static,
            segregating=segregating,
            orientation=transpose,
            diagnostics=diagnostics,
            throw=throw,
            cached=cached,
        )
        super().__init__()

    def _config(self):
        return dict(
            segregating=self.params["segregating"],
            transpose=self.params["orientation"],
            diagnostics=self.params["diagnostics"],
            throw=self.params["throw"],
        )

    def _primal(self, model, state, rate, output):
        return _bind_action(model, state, rate, output, **self._config())

    def expand(self, *args):
        """Evaluate the checked action, or reuse an already computed primal."""
        model, state, rate, *cached = eqx.combine(
            args, cast(tuple, self.params["static"])
        )
        if self.params["cached"]:
            return cached[0]
        value, auxiliary = _evaluate_with_aux(model, state, rate, **self._config())
        return _pack(value, auxiliary, self.params["diagnostics"])

    def jvp(self, primals, tangents):
        """Evaluate the common action derivative alongside its primal once."""
        model, state, rate, *_ = eqx.combine(
            primals, cast(tuple, self.params["static"])
        )
        t_model, t_state, t_rate, *_ = tangents
        if _has_tangent((t_model.geometry, t_model.method)):
            raise TypeError(
                "demographic geometry and numerical method configuration are nondifferentiable"
            )
        directions = jax.tree.map(
            _tangent_or_none, (t_model, t_state, t_rate), is_leaf=_is_ad_leaf
        )

        def action(model, state, rate):
            value, auxiliary = _evaluate_with_aux(model, state, rate, **self._config())
            return _pack(value, auxiliary, self.params["diagnostics"])

        value, tangent = eqx.filter_jvp(action, (model, state, rate), directions)
        tangent = jax.tree.map(
            lambda aval, diff: (
                hijax.Zero(aval.to_tangent_aval()) if diff is None else diff
            ),
            self.out_aval,
            tangent,
            is_leaf=lambda x: x is None,
        )
        return self._primal(model, state, rate, value), tangent

    lin, linearized = hijax.linearize_from_jvp

    def vjp_fwd(self, nzs, *primals):
        """Retain parameter residuals from the original oriented action."""
        _check_fixed(nzs[0])
        model, state, rate, *_ = eqx.combine(
            primals, cast(tuple, self.params["static"])
        )
        mask = _parameter_mask(nzs)
        pullback = None
        if any(jax.tree.leaves(mask)):
            value, pullback, auxiliary = _parameter_vjp(
                model,
                state,
                rate,
                mask,
                segregating=self.params["segregating"],
                transpose=self.params["orientation"],
                diagnostics=self.params["diagnostics"],
                throw=self.params["throw"],
            )
        else:
            value, auxiliary = _evaluate_with_aux(model, state, rate, **self._config())
        output = _pack(value, auxiliary, self.params["diagnostics"])
        return self._primal(model, state, rate, output), (
            primals,
            pullback,
            eqxi.Static(nzs),
        )

    def vjp_bwd_retval(self, residuals, out_ct):
        """Use the opposite vector action and the shared parameter pairing."""
        primals, pullback, static_nzs = residuals
        state_ct = out_ct[0] if self.params["diagnostics"] else out_ct
        # Unused action outputs have structural zeros. Return without allocating
        # zero arrays or invoking numerical pullbacks; diagnostic cotangents do
        # not contribute. Numerical zeros must retain their higher derivatives.
        if not _has_tangent(state_ct):
            return jax.tree.map(
                lambda aval: hijax.Zero(aval.to_tangent_aval()), self.in_avals
            )
        nzs = static_nzs.value
        model, state, rate, *_ = eqx.combine(
            primals, cast(tuple, self.params["static"])
        )
        cotangent = jax.tree.map(hijax.instantiate_zeros, state_ct, is_leaf=_is_ad_leaf)
        model_grad = rate_grad = state_grad = None
        mask = _parameter_mask(nzs)
        if any(jax.tree.leaves(mask)):
            model_grad, rate_grad = pullback(cotangent)[0]
        if any(jax.tree.leaves(nzs[1])):
            state_grad = _bind_action(
                model,
                cotangent,
                rate,
                segregating=self.params["segregating"],
                transpose=not self.params["orientation"],
            )
        # Return symbolic zeros for inactive dynamic leaves, including cached
        # outputs and diagnostic-only dependencies. Static leaves remain None.
        grads = (model_grad, state_grad, rate_grad, *([None] * (len(primals) - 3)))

        def restore(primal, gradient):
            if primal is None:
                return None
            if gradient is not None and eqx.is_array(primal):
                return gradient
            if gradient is None:
                return jax.tree.map(
                    lambda x: hijax.Zero(jax.typeof(x).to_tangent_aval()), primal
                )
            return jax.tree.map(restore, primal, gradient, is_leaf=lambda x: x is None)

        return tuple(restore(p, g) for p, g in zip(primals, grads, strict=True))

    def transpose(self, out_ct, *args):
        """Transpose the augmented vector through the opposite action."""
        model, state, rate, *_ = eqx.combine(args, cast(tuple, self.params["static"]))
        if (
            self.params["cached"]
            or self.params["diagnostics"]
            or _has_accum((model, rate))
        ):
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
        """Map the state and diagnostic outputs with mapped mathematical inputs."""
        dim = 0 if any(d is not None for d in jax.tree.leaves(dims)) else None
        return jax.tree.map(lambda _: dim, self.out_aval)


def _bind_action(
    model, state, rate, *cached, segregating, transpose, diagnostics=False, throw=True
):
    """Bind the mathematical inputs and optional checked primal output."""
    dynamic, static = eqx.partition((model, state, rate, *cached), eqx.is_array)
    avals = jax.tree.map(jax.typeof, dynamic)
    return _DemographicAction(
        avals,
        static,
        segregating=segregating,
        transpose=transpose,
        diagnostics=diagnostics,
        throw=throw,
        cached=bool(cached),
    )(*dynamic)


def _demographic_advance(model, distribution, *, segregating, singleton_rate, throw):
    """Reconstruct public diagnostics from a single checked primitive evaluation."""
    (mass, _), (status, intervals) = _bind_action(
        model,
        (distribution, jnp.asarray(1.0, dtype=jnp.float64)),
        singleton_rate,
        segregating=segregating,
        transpose=True,
        diagnostics=True,
        throw=throw,
    )
    diagnostics = tuple(
        IntervalDiagnostics(
            source,
            destination,
            duration,
            error,
            interval_status == EvolutionStatus.successful,
            interval_status,
        )
        for source, destination, duration, (error, interval_status) in zip(
            model.demography.pop_sizes[:-1],
            model.demography.pop_sizes[1:],
            model.demography.generations,
            intervals,
            strict=True,
        )
    )
    return DemographicResult(
        mass, status == EvolutionStatus.successful, status, diagnostics
    )


def _demographic_expect(model, values, *, segregating, singleton_rate):
    """Apply the backward action with zero initial source reward."""
    return _bind_action(
        model,
        (values, jnp.asarray(0.0, dtype=jnp.float64)),
        singleton_rate,
        segregating=segregating,
        transpose=False,
    )
