"""Population histories and their demographic transitions."""

# pattern: Functional Core
import operator
from collections.abc import Callable, Iterable
from typing import NamedTuple, final

import equinox as eqx
import jax
import jax.numpy as jnp

from ._backend import AbstractTransitionBackend
from ._demographic_action import _demographic_advance, _demographic_expect
from ._evolution.methods import AbstractEpochMethod, Sequential
from ._evolution.preparation import validate_domain
from ._evolution.problem import EvolutionStatus, _epoch_result
from ._frequency import AbstractFrequencyMap
from ._operator import Transition


def _singleton_inputs(vector, segregating, singleton_mutation_rate):
    """Normalize a vector and validate the shared singleton-source contract."""
    if not isinstance(segregating, bool):
        raise ValueError("segregating must be a Python bool")
    if not segregating and singleton_mutation_rate is not None:
        raise ValueError("singleton_mutation_rate requires segregating=True")
    vector = jnp.asarray(vector, dtype=jnp.float64)
    rate = jnp.asarray(
        0.0 if singleton_mutation_rate is None else singleton_mutation_rate,
        dtype=jnp.float64,
    )
    if rate.shape != ():
        raise ValueError("singleton_mutation_rate must be scalar")
    vector = eqx.error_if(
        vector,
        ~jnp.isfinite(rate) | (rate < 0),
        "singleton_mutation_rate must be finite and nonnegative",
    )
    return vector, rate


def _integer_schedule(values: Iterable[int] | jax.Array) -> tuple[int, ...]:
    """Copy concrete integer schedule entries to immutable host metadata."""
    result = []
    try:
        for value in values:
            if isinstance(value, jax.Array):
                if value.ndim != 0:
                    raise ValueError("schedule entries must be scalars")
                value = value.item()
            if isinstance(value, bool):
                raise ValueError("schedule entries must be integers, not booleans")
            result.append(operator.index(value))
    except (TypeError, OverflowError) as error:
        raise ValueError(
            "schedule must be a one-dimensional sequence of integers"
        ) from error
    return tuple(result)


@final
class Demography(eqx.Module):
    r"""A population history specified at successive generation boundaries.

    Sizes run from oldest to newest. Each interval holds the source size for
    `generations[i] - 1` generations, then changes size on the final generation.
    Repeated sizes are valid; a single size with no durations is an identity
    history. No equilibrium initialization is implied. The final output is taken
    immediately after the last size-changing step, without extra resampling or
    evolution at the final size.
    """

    pop_sizes: tuple[int, ...] = eqx.field(converter=_integer_schedule)
    generations: tuple[int, ...] = eqx.field(converter=_integer_schedule)

    def __check_init__(self) -> None:
        """Validate chronological schedule dimensions and positive counts."""
        if not self.pop_sizes or len(self.pop_sizes) != len(self.generations) + 1:
            raise ValueError(
                "pop_sizes must be nonempty and have len(generations) + 1 entries"
            )
        if any(size < 1 for size in self.pop_sizes):
            raise ValueError("pop_sizes entries must be positive integers")
        if any(duration < 1 for duration in self.generations):
            raise ValueError("generations entries must be positive integers")

    def prepare(
        self,
        *,
        backend_factory: Callable[[int], AbstractTransitionBackend],
        boundary_backend_factory: Callable[[int], AbstractTransitionBackend]
        | None = None,
    ) -> "DemographicGeometry":
        """Prepare reusable geometry outside JIT, independently of the process.

        **Arguments:**

        - `backend_factory`: Factory receiving each distinct diploid population
          size. Use `functools.partial` to supply backend settings.
        - `boundary_backend_factory`: Optional factory for rectangular boundary
          transitions. Defaults to the hold factory, sharing its geometry.

        **Returns:**

        A [`DemographicGeometry`][drifft.DemographicGeometry] reusable across frequency maps and epoch methods.
        The population schedule is unchanged. Parameter-dependent operators are
        assembled when computing with a bound [`DemographicTransition`][drifft.DemographicTransition].

        **Raises:**

        `TypeError` for invalid factories or returned backend types;
        `ValueError` when a backend does not preserve the requested size.
        """
        return DemographicGeometry(
            self,
            backend_factory=backend_factory,
            boundary_backend_factory=boundary_backend_factory,
        )

    @property
    def initial_pop_size(self) -> int:
        r"""Initial diploid effective population size, `pop_sizes[0]`, as a Python integer."""
        return self.pop_sizes[0]

    @property
    def final_pop_size(self) -> int:
        r"""Final diploid effective population size, `pop_sizes[-1]`, as a Python integer."""
        return self.pop_sizes[-1]

    @property
    def total_generations(self) -> int:
        r"""Total interval duration as a Python integer; zero for an identity history."""
        return sum(self.generations)


# Document the generated constructor without replacing its implementation.
Demography.__init__.__doc__ = r"""Initialize [`Demography`][drifft.Demography].

**Arguments:**

- `pop_sizes`: Nonempty sequence of positive integer diploid effective population
  sizes, ordered oldest to newest. Size `pop_sizes[i]` represents
  `2 * pop_sizes[i]` allele-bearing copies.
- `generations`: Positive integer durations between successive sizes, with
  exactly one fewer entry than `pop_sizes`.

Lists and concrete one-dimensional integer arrays are converted to host
tuples. Construct outside JIT.

**Raises:**

`ValueError` for invalid lengths, noninteger entries, or nonpositive entries.
Population sizes and durations are structural, not differentiable parameters.
"""


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


@final
class DemographicGeometry(eqx.Module):
    """Prepared population-size backends and their immutable demographic schedule.

    Construct with [`Demography.prepare`][drifft.Demography.prepare] outside JIT. Geometry contains no bound
    frequency map or epoch method and can be shared across parameter evaluations.
    Backend approximation domains still constrain the processes used with it.
    """

    demography: Demography
    _backends: dict[int, AbstractTransitionBackend]
    _boundary_backends: dict[int, AbstractTransitionBackend]

    def __init__(
        self,
        demography: Demography,
        *,
        backend_factory: Callable[[int], AbstractTransitionBackend],
        boundary_backend_factory: Callable[[int], AbstractTransitionBackend]
        | None = None,
    ) -> None:
        """Prepare one backend per distinct size; prefer [`Demography.prepare`][drifft.Demography.prepare]."""
        if not isinstance(demography, Demography):
            raise TypeError("demography must be a Demography")
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable")
        if boundary_backend_factory is not None and not callable(
            boundary_backend_factory
        ):
            raise TypeError("boundary_backend_factory must be callable")
        self.demography = demography
        self._backends = {
            size: backend_factory(size) for size in dict.fromkeys(demography.pop_sizes)
        }
        for size, geometry in self._backends.items():
            if not isinstance(geometry, AbstractTransitionBackend):
                raise TypeError(
                    "backend_factory must return an AbstractTransitionBackend"
                )
            if geometry.pop_size != size:
                raise ValueError(
                    "backend_factory must preserve the requested diploid size"
                )
        if (
            boundary_backend_factory is None
            or boundary_backend_factory is backend_factory
        ):
            self._boundary_backends = self._backends
        else:
            self._boundary_backends = {
                size: boundary_backend_factory(size) for size in self._backends
            }

    def __check_init__(self) -> None:
        """Require rectangular transition geometry at all boundaries."""
        for size, backend in self._boundary_backends.items():
            if not isinstance(backend, AbstractTransitionBackend):
                raise TypeError(
                    "boundary_backend_factory must return an AbstractTransitionBackend"
                )
            if backend.pop_size != size:
                raise ValueError(
                    "boundary_backend_factory must preserve the requested diploid size"
                )


@final
class DemographicTransition(eqx.Module):
    r"""Bind prepared demographic geometry to a differentiable frequency map.

    Let $P_t$ be the one-generation kernel from diploid size $N_t$ to
    $N_{t+1}$, using the shared frequency map and the binomial law defined by
    [`Transition`][drifft.Transition]. Each $P_t$ has shape
    $(2N_t+1)\times(2N_{t+1}+1)$. The whole-history operator is

    $$ H = P_0 P_1 \cdots P_{T-1}, $$

    where $T$ is the total number of generations. Without source input,

    $$ p_T = H^T p_0, \qquad v_0 = Hf, $$

    for initial mass $p_0$ and terminal values $f$. An interval from size $N$
    to $N'$ lasting $g$ generations contributes

    $$ P_{N\to N}^{\,g-1} P_{N\to N'}. $$

    Thus the source size is held for $g-1$ generations, and the size change
    occurs on the final generation. An empty history has $H=I$. Segregating
    operations use the interior restrictions of these kernels.

    Numerical methods evaluate or approximate the action without forming $H$.
    Both orientations use the configured method. Krylov forward and backward
    actions approximate the same represented operator independently. The method
    docstrings below describe singleton source input and its expectation offset.

    Differentiation targets these mathematical actions with respect to process
    parameters, input mass or values, and singleton rate. Geometry and method
    settings are fixed. [`Sequential`][drifft.Sequential] supports higher
    derivatives through the represented transitions; Krylov operator
    sensitivities support first derivatives. Adaptive projection bases and
    numerical diagnostics are not differentiation targets.
    """

    geometry: DemographicGeometry
    frequency_map: AbstractFrequencyMap
    method: AbstractEpochMethod

    def __init__(
        self,
        geometry: DemographicGeometry,
        frequency_map: AbstractFrequencyMap,
        *,
        method: AbstractEpochMethod | None = None,
    ) -> None:
        """Bind a process and algorithm without rebuilding geometry.

        This lightweight constructor may be called inside JAX transformations.
        Prepare geometry once with [`Demography.prepare`][drifft.Demography.prepare] outside JIT, then build
        the frequency map from the current selection and mutation parameters.

        **Arguments:**

        - `geometry`: Prepared demographic geometry, including the schedule.
        - `frequency_map`: Process shared by every generation.
        - `method`: Geometry-free epoch algorithm; defaults to [`Sequential()`][drifft.Sequential].
          Holds call its `advance` method, including custom overrides.

        **Raises:**

        `TypeError` for invalid geometry, frequency maps, methods, or unsupported
        method/backend/process combinations, including histories without holds.
        """
        if not isinstance(geometry, DemographicGeometry):
            raise TypeError(
                "geometry must be a DemographicGeometry; use Demography.prepare"
            )
        if not isinstance(frequency_map, AbstractFrequencyMap):
            raise TypeError(
                "frequency_map must be an AbstractFrequencyMap; wrap functions in FrequencyMap"
            )
        self.geometry = geometry
        self.frequency_map = frequency_map
        self.method = Sequential() if method is None else method
        if not isinstance(self.method, AbstractEpochMethod):
            raise TypeError("method must be an AbstractEpochMethod")
        for backend in geometry._backends.values():
            self.method.validate(backend, frequency_map)

    @property
    def demography(self) -> Demography:
        """The population schedule associated with the prepared geometry."""
        return self.geometry.demography

    def _transition(self, source_pop_size: int, dest_pop_size: int) -> Transition:
        return Transition(
            self.geometry._boundary_backends[dest_pop_size],
            self.frequency_map,
            source_pop_size=source_pop_size,
        )

    @eqx.filter_jit
    def advance(
        self,
        distribution: jax.Array,
        *,
        segregating: bool = False,
        singleton_mutation_rate: float | jax.Array | None = None,
        throw: bool = True,
    ) -> DemographicResult:
        r"""Propagate mass through the history, optionally discarding endpoint hits.

        Let $P_t$ map source counts at generation $t$ to destination counts at
        $t+1$, including any population-size change. Over $T$ total generations,
        the forward recurrence is

        $$ p_{t+1} = P_t^T p_t + b_{t+1}, \qquad t=0,\ldots,T-1. $$

        By default $b_{t+1}=0$. With singleton input at per-copy rate $\mu$,
        $b_{t+1}=2N_{t+1}\mu e_1$, where $N_{t+1}$ is the destination diploid
        size and $e_1$ places mass at count one after reproduction. This source
        requires `segregating=True`. In segregating mode, replace $P_t$ with its
        interior restriction; endpoint mass is discarded without conditioning
        on survival. Epoch methods evaluate or approximate this evolution.

        !!! example "Example: Follow a singleton through a population history"

            Given a bound demographic transition `model`:

            ```python
            # Start with one derived copy on the history's initial population grid.
            initial = jnp.zeros(2 * model.demography.initial_pop_size + 1).at[1].set(1.0)
            result = model.advance(initial)
            # The final grid can have a different size; index zero still means absence.
            final_distribution = result.distribution
            absent_probability = final_distribution[0]
            ```

        **Arguments:**

        - `distribution`: Real mass of length `2 * initial_pop_size + 1`, or
          `2 * initial_pop_size - 1` when segregating. Converted to float64; signed
          entries are supported.
        - `segregating`: Discard endpoint mass after each generation.
        - `singleton_mutation_rate`: Finite nonnegative scalar per-copy rate,
          allowed only when segregating; defaults to zero. Adds
          `2 * dest_pop_size * singleton_mutation_rate` at count one after
          every generation. Recurrent mutation belongs in the frequency map.
        - `throw`: Raise on numerical failure; under vmap may fail the whole batch.

        **Returns:**

        [`DemographicResult`][drifft.DemographicResult] containing final mass and each interval's outcome.
        Output length is `2 * final_pop_size + 1`, or `2 * final_pop_size - 1` when segregating.
        No mass normalization occurs. Derivatives of `distribution` include
        process, input-mass, and singleton-rate dependence; diagnostics are
        excluded from differentiation.

        **Raises:**

        `ValueError` for invalid shapes, flags, or singleton input in full-state
        mode. Invalid rates or process domains raise even with `throw=False`.
        Primal numerical failures raise with `throw=True` and return status
        with false. Sensitivity and pullback failures raise even with
        `throw=False`. Higher-order Krylov operator sensitivities raise explicitly.
        """
        distribution, rate = _singleton_inputs(
            distribution, segregating, singleton_mutation_rate
        )
        return _demographic_advance(
            self,
            distribution,
            segregating=segregating,
            singleton_rate=rate,
            throw=throw,
        )

    def _advance(self, distribution, *, segregating, singleton_rate, throw):
        distribution = jnp.asarray(distribution, dtype=jnp.float64)
        size = 2 * self.demography.initial_pop_size + (-1 if segregating else 1)
        if distribution.shape != (size,):
            raise ValueError(f"distribution must have {size} entries")
        # Validate every prepared domain even for identity/boundary-only histories.
        for backend in self.geometry._backends.values():
            distribution = validate_domain(
                backend,
                distribution,
                self.frequency_map,
                reduced=self.method.allows_reduced_representation,
            )
        initial_result = _epoch_result(distribution, jnp.asarray(0.0), 0.0, False)
        status = initial_result.status
        diagnostics = []
        for source, destination, duration in zip(
            self.demography.pop_sizes[:-1],
            self.demography.pop_sizes[1:],
            self.demography.generations,
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
                result = self.method.advance(
                    self.geometry._backends[source],
                    distribution,
                    frequency_map=self.frequency_map,
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
            transition = self._transition(source, destination)
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

    @eqx.filter_jit
    def expect(
        self,
        values: jax.Array,
        *,
        segregating: bool = False,
        singleton_mutation_rate: float | jax.Array | None = None,
    ) -> DemographicExpectation:
        r"""Compute the expectation action of the demographic history.

        Let $P_t$ be the one-generation transition from $t$ to $t+1$, including
        size changes, and $b_{t+1}$ the source added after that step. Starting
        from terminal values $v_T=f$ and $c_T=0$, work backward through the
        $T$ generations:

        $$ v_t = P_t v_{t+1}, \qquad
           c_t = c_{t+1} + b_{t+1}^T v_{t+1}. $$

        The returned `values` and `offset` are $v_0$ and $c_0$, satisfying

        $$ p_T^T f = p_0^T v_0 + c_0. $$

        The source is zero by default. With singleton input,
        $b_{t+1}=2N_{t+1}\mu e_1$, using the destination diploid size and
        per-copy rate. In segregating mode, $P_t$ denotes its interior
        restriction, discarding endpoint mass without conditioning on survival.

        !!! example "Example: Expected final allele count"

            ```python
            # Define the observable on the final grid, where the allele is counted.
            counts = jnp.arange(2 * model.demography.final_pop_size + 1)
            expectation = model.expect(counts)
            # Each returned value corresponds to an initial population count.
            initial = jnp.zeros(2 * model.demography.initial_pop_size + 1).at[1].set(1.0)
            # Include the affine offset; it is zero here because no source was supplied.
            expected_count = initial @ expectation.values + expectation.offset
            ```

        !!! warning "Krylov approximation"

            Krylov methods approximate the backward action independently of
            the forward action. Their pairing agrees to numerical accuracy,
            rather than being an exact transpose of an input-dependent
            projection. Increase dimension or tighten tolerances as needed.

        **Arguments:**

        - `values`: Real vector of length `2 * final_pop_size + 1`, or
          `2 * final_pop_size - 1` when segregating, converted to float64.
        - `segregating`: Discard endpoint mass after each generation.
        - `singleton_mutation_rate`: Finite nonnegative scalar per-copy source
          rate, as in `advance`. Requires segregating states.
          Recurrent mutation remains part of the frequency map.

        **Returns:**

        A [`DemographicExpectation`][drifft.DemographicExpectation] with float64 `values` of length
        `2 * initial_pop_size + 1`, or `2 * initial_pop_size - 1` when segregating, and a
        scalar `offset`. The reward `initial @ result.values + result.offset`
        equals the terminal reward from `advance` with the same layout and rate,
        up to numerical approximation error.
        The offset is zero without a source or for an identity history.
        Both `values` and `offset` retain their process and input-value
        derivatives. Only `offset` depends on singleton rate; its rate
        derivative can be nonzero even at zero rate.
        Signed vectors are supported; no normalization occurs.

        **Raises:**

        `ValueError` for incompatible shapes, nonboolean flags, invalid source
        configuration. Numerical convergence and sensitivity failures raise at
        runtime. Higher-order Krylov operator sensitivities raise explicitly.
        Invalid rate values raise at runtime, including for identity histories.
        Input finiteness is a caller
        precondition, as for [`Transition.expect`][drifft.Transition.expect].
        """
        values, rate = _singleton_inputs(values, segregating, singleton_mutation_rate)
        size = 2 * self.demography.final_pop_size + (-1 if segregating else 1)
        if values.shape != (size,):
            raise ValueError(f"values must have {size} entries")
        expected, offset = _demographic_expect(
            self, values, segregating=segregating, singleton_rate=rate
        )
        # Keep validation live even when an outer JIT consumes only the
        # identity history's offset and discards the expectation vector.
        return eqx.error_if(
            DemographicExpectation(expected, offset),
            ~jnp.isfinite(rate) | (rate < 0),
            "singleton_mutation_rate must be finite and nonnegative",
        )
