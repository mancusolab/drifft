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
representation. Its implementation composes epoch powers and population-size
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

from ._evolution.autodiff import (
    _accumulate,
    _has_accum,
    _has_tangent,
    _is_ad_leaf,
    _tangent_or_none,
)
from ._evolution.methods import prepare_epoch
from ._evolution.preparation import validate_domain
from ._evolution.problem import EvolutionStatus, _epoch_result, _working_problem
from ._operator import Transition


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


def _transition(model, source_pop_size: int, dest_pop_size: int) -> Transition:
    """Bind a boundary transition using the destination's prepared backend.

    The source size sets the input grid and the destination backend sets the
    output grid. Both history orientations use this same represented kernel,
    with the bound frequency map retaining its parameter derivatives.
    """
    return Transition(
        model.geometry._boundary_backends[dest_pop_size],
        model.frequency_map,
        source_pop_size=source_pop_size,
    )


def _advance_history(model, distribution, *, segregating, singleton_rate, throw):
    """Propagate mass forward through holds and boundaries, retaining diagnostics.

    Each interval reserves its final generation for the rectangular boundary
    transition. Singleton input is added after each generation at that
    generation's destination size. This is the forward numerical traversal
    paired with `_expect_history`; the enclosing primitive supplies its AD rules.
    """
    from ._demography import DemographicResult, IntervalDiagnostics

    distribution = jnp.asarray(distribution, dtype=jnp.float64)
    size = 2 * model.demography.initial_pop_size + (-1 if segregating else 1)
    if distribution.shape != (size,):
        raise ValueError(f"distribution must have {size} entries")
    # Validate reduced domains only for the prepared constant-size holds.
    for size in model.geometry:
        distribution = validate_domain(
            model.geometry[size],
            distribution,
            model.frequency_map,
            reduced=model.method.allows_reduced_representation,
        )
    initial_result = _epoch_result(distribution, jnp.asarray(0.0), 0.0, False)
    status = initial_result.status
    diagnostics = []
    for source, destination, duration in zip(
        model.demography.pop_sizes[:-1],
        model.demography.pop_sizes[1:],
        model.demography.generations,
        strict=True,
    ):
        # The final generation belongs to the rectangular boundary action.
        # Holding for duration instead would add an extra generation and,
        # in segregating mode, an extra source contribution at the source size.
        error = jnp.asarray(0.0)
        interval_status = EvolutionStatus.successful
        if duration > 1:
            source_mass = (
                jnp.zeros_like(distribution).at[0].set(2 * source * singleton_rate)
                if segregating
                else None
            )
            result = model.method.advance(
                model.geometry[source],
                distribution,
                frequency_map=model.frequency_map,
                generations=duration - 1,
                segregating=segregating,
                source_mass=source_mass,
                throw=False,
            )
            distribution, error, interval_status = (
                result.distribution,
                result.error_estimate,
                result.status,
            )
        transition = _transition(model, source, destination)
        distribution = transition.advance(distribution, segregating=segregating)
        if segregating:
            distribution = distribution.at[0].add(2 * destination * singleton_rate)
        boundary = _epoch_result(distribution, jnp.asarray(0.0), 0.0, False)
        interval_status = EvolutionStatus.where(
            interval_status == EvolutionStatus.successful,
            boundary.status,
            interval_status,
        )
        status = EvolutionStatus.where(
            status == EvolutionStatus.successful, interval_status, status
        )
        diagnostics.append(
            IntervalDiagnostics(
                source,
                destination,
                duration,
                error,
                interval_status == EvolutionStatus.successful,
                interval_status,
            )
        )
    converged = status == EvolutionStatus.successful
    if throw:
        distribution = eqx.error_if(
            distribution,
            ~converged,
            "demographic evolution failed: inspect interval diagnostics with throw=False; "
            "check input mass and increase dimension or use Sequential",
        )
    return DemographicResult(distribution, converged, status, tuple(diagnostics))


def _expect_history(model, state, rate, *, segregating):
    """Apply (f, a) -> (H f, a + c.T f) by reversing the history.

    The scalar coordinate is arbitrary in pullbacks; public expect supplies
    zero. Epoch primitives provide mathematical derivatives without AD through
    adaptive projection machinery.
    """
    values, offset = state
    for size in model.geometry:
        values = validate_domain(
            model.geometry[size],
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
        values = _transition(model, source, destination).expect(
            values, segregating=segregating
        )
        if duration > 1:
            values, contribution = _expect_epoch(
                model,
                model.geometry[source],
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
