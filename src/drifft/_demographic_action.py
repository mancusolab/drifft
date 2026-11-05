"""Mathematical demographic actions and their shared HiJAX differentiation rules."""

# pattern: Functional Core
from typing import cast

import equinox as eqx
import equinox.internal as eqxi
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
from ._evolution.problem import EvolutionStatus, _working_problem


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

    def prepare_initial(mass):
        problem = prepare_epoch(
            model.method,
            backend,
            mass,
            frequency_map=model.frequency_map,
            generations=generations,
            segregating=segregating,
            source_mass=source,
        )
        return problem.initial, problem

    # Return the prepared problem as auxiliary data so encoding and its
    # transpose share one preparation. Outer process derivatives remain live.
    seed, encode_transpose, problem = eqx.filter_vjp(
        prepare_initial, zero, has_aux=True
    )
    _, decode_transpose = jax.vjp(problem.decode, jnp.zeros_like(problem.initial))
    terminal = decode_transpose(values)[0]
    working = model.method.solve(
        _working_problem(problem.operator.T, terminal, problem.steps)
    ).distribution
    offset = seed @ working + problem.output_offset @ values
    return encode_transpose(working)[0], offset


def _expect_history(model, state, rate, *, segregating):
    """Apply (f, a) -> (H f, a + c.T f) by reversing the history.

    The scalar coordinate is arbitrary in pullbacks; public expect supplies
    zero. Epoch primitives provide mathematical derivatives without AD through
    adaptive projection machinery.
    """
    values, offset = state
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


def _evaluate_with_aux(
    model, state, rate, *, segregating, transpose, diagnostics, throw
):
    """Evaluate once and separate nondifferentiable numerical diagnostics."""
    if transpose:
        vector, scalar = state
        result = model._advance(
            vector, segregating=segregating, singleton_rate=rate * scalar, throw=throw
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
    model, seed, rate, mask, *, segregating, diagnostics=False, throw=True
):
    """Retain the forward scalar-pairing pullback and its numerical intermediates.

    The forward action uses its original seed. For the backward action the
    incoming cotangent becomes the seed and the original input becomes the
    reward. Both orientations therefore differentiate the same pairing.
    """
    active, fixed = eqx.partition((model, rate), mask)

    def forward(parameters):
        current_model, current_rate = eqx.combine(parameters, fixed)
        return _evaluate_with_aux(
            current_model,
            seed,
            current_rate,
            segregating=segregating,
            transpose=True,
            diagnostics=diagnostics,
            throw=throw,
        )

    return eqx.filter_vjp(forward, active, has_aux=True)


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
    """An oriented history action that reuses its primal work in differentiation.

    JVPs evaluate the primal and direction together. Forward parameter VJPs
    retain the epoch linearization from that same primal evaluation; vector
    cotangents invoke the opposite action. Cached outputs preserve this boundary
    under higher derivatives without repeating the primal calculation.
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
        """Retain the forward parameter linearization instead of recomputing it."""
        _check_fixed(nzs[0])
        model, state, rate, *_ = eqx.combine(
            primals, cast(tuple, self.params["static"])
        )
        mask = _parameter_mask(nzs)
        pullback = None
        if self.params["orientation"] and any(jax.tree.leaves(mask)):
            value, pullback, auxiliary = _parameter_vjp(
                model,
                state,
                rate,
                mask,
                segregating=self.params["segregating"],
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
        opposite_value = None
        mask = _parameter_mask(nzs)
        if any(jax.tree.leaves(mask)):
            if self.params["orientation"]:
                model_grad, rate_grad = pullback(cotangent)[0]
            else:
                opposite_value, backward_pairing, _ = _parameter_vjp(
                    model,
                    cotangent,
                    rate,
                    mask,
                    segregating=self.params["segregating"],
                )
                model_grad, rate_grad = backward_pairing(state)[0]
        if any(jax.tree.leaves(nzs[1])):
            # For expect, the forward pairing already produced A @ cotangent.
            # Keep the opposite primitive boundary for higher derivatives,
            # using that value instead of evaluating the action a second time.
            cached = () if opposite_value is None else (opposite_value,)
            state_grad = _bind_action(
                model,
                cotangent,
                rate,
                *cached,
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
    from ._demography import DemographicResult, IntervalDiagnostics

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
