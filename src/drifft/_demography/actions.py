r"""Forward and backward history traversal with numerical result containers.

Compose constant-size epochs and rectangular boundary transitions, including
singleton sources and per-interval diagnostics. These routines implement the
history actions; autodiff.py wraps them with mathematical derivative rules.
They consume the bound model without importing its public transition class.
"""

# pattern: Functional Core
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .._evolution.methods import prepare_epoch
from .._evolution.preparation import validate_domain
from .._evolution.problem import EvolutionStatus, _epoch_result, _working_problem
from .._operator import Transition


class IntervalDiagnostics(NamedTuple):
    r"""Numerical outcome for one complete demographic interval.

    `source_pop_size` and `dest_pop_size` are diploid population sizes; `generations`
    includes both the hold and final boundary generation. `error_estimate`
    covers only the hold power approximation, while `converged` and `status`
    also account for nonfinite boundary output.

    **Fields:**

    - `source_pop_size`, `dest_pop_size`: Python integers giving diploid sizes.
    - `generations`: Interval duration, including its final boundary generation.
    - `error_estimate`: Scalar absolute L1 hold-power estimate, zero without a hold.
    - `converged`: Scalar boolean for the complete interval.
    - `status`: First hold or boundary failure, or successful [`EvolutionStatus`][drifft.EvolutionStatus].
    """

    source_pop_size: int
    dest_pop_size: int
    generations: int
    error_estimate: jax.Array
    converged: jax.Array
    status: EvolutionStatus


# Document the generated constructor without replacing its implementation.
IntervalDiagnostics.__new__.__module__ = __name__
IntervalDiagnostics.__new__.__doc__ = r"""Construct an immutable result without validation.

**Arguments:**

The fields documented on [`IntervalDiagnostics`][drifft.IntervalDiagnostics], supplied by position or keyword.

**Returns:**

An [`IntervalDiagnostics`][drifft.IntervalDiagnostics] holding the supplied values. Normal evolution calls construct this
result automatically; constructing it directly performs no numerical checks.
"""


class DemographicResult(NamedTuple):
    r"""Final mass and per-interval numerical outcomes, without a total error certificate.

    `status` retains the first numerical failure. Identity histories have empty
    `diagnostics` and still check mass finiteness. No factorization or basis is retained.

    **Fields:**

    - `distribution`: Float64 mass of length `2 * final_pop_size + 1`, or
      `2 * final_pop_size - 1` for segregating evolution; not normalized.
    - `converged`: Scalar boolean, true when the entire history succeeded.
    - `status`: First numerical failure in chronological order, or success.
    - `diagnostics`: Tuple of [`IntervalDiagnostics`][drifft.IntervalDiagnostics], one per interval in order.
      Local error estimates are not a bound on the total propagated error.
    """

    distribution: jax.Array
    converged: jax.Array
    status: EvolutionStatus
    diagnostics: tuple[IntervalDiagnostics, ...]


# Document the generated constructor without replacing its implementation.
DemographicResult.__new__.__module__ = __name__
DemographicResult.__new__.__doc__ = r"""Construct an immutable result without validation.

**Arguments:**

The fields documented on [`DemographicResult`][drifft.DemographicResult], supplied by position or keyword.

**Returns:**

A [`DemographicResult`][drifft.DemographicResult] holding the supplied values. Normal evolution calls construct this
result automatically; constructing it directly performs no numerical checks.
"""


class DemographicExpectation(NamedTuple):
    r"""Backward values and the reward contributed by singleton mutation input.

    For initial mass `initial`, the terminal reward is
    `initial @ result.values + result.offset`.

    **Fields:**

    - `values`: Float64 expectation vector on the initial population's full or
      interior grid. Independent of the singleton source rate.
    - `offset`: Float64 scalar reward from singleton input across the history.
      Zero without a source or for an identity history.
    """

    values: jax.Array
    offset: jax.Array


DemographicExpectation.__new__.__module__ = __name__
DemographicExpectation.__new__.__doc__ = """Construct an immutable result without validation.

**Arguments:**

The fields documented on [`DemographicExpectation`][drifft.DemographicExpectation], supplied by position or keyword.

**Returns:**

A [`DemographicExpectation`][drifft.DemographicExpectation] holding the supplied values.
"""


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
