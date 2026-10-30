"""Population histories and their demographic transitions."""

# pattern: Functional Core
import operator
from collections.abc import Callable, Iterable
from typing import NamedTuple, final

import equinox as eqx
import jax
import jax.numpy as jnp

from ._backend import AbstractTransitionBackend
from ._evolution.methods import AbstractEpochMethod, Sequential, prepare_epoch
from ._evolution.preparation import validate_domain
from ._evolution.problem import EvolutionStatus, _epoch_result
from ._frequency import AbstractFrequencyMap
from ._operator import Transition


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
    history. No equilibrium initialization is implied.
    """

    ne: tuple[int, ...] = eqx.field(converter=_integer_schedule)
    generations: tuple[int, ...] = eqx.field(converter=_integer_schedule)

    def __check_init__(self) -> None:
        """Validate chronological schedule dimensions and positive counts."""
        if not self.ne or len(self.ne) != len(self.generations) + 1:
            raise ValueError(
                "ne must be nonempty and have len(generations) + 1 entries"
            )
        if any(size < 1 for size in self.ne):
            raise ValueError("ne entries must be positive integers")
        if any(duration < 1 for duration in self.generations):
            raise ValueError("generations entries must be positive integers")

    @property
    def initial_ne(self) -> int:
        r"""Initial diploid effective population size, `ne[0]`, as a Python integer."""
        return self.ne[0]

    @property
    def final_ne(self) -> int:
        r"""Final diploid effective population size, `ne[-1]`, as a Python integer."""
        return self.ne[-1]

    @property
    def total_generations(self) -> int:
        r"""Total interval duration as a Python integer; zero for an identity history."""
        return sum(self.generations)


# Document the generated constructor without replacing its implementation.
Demography.__init__.__doc__ = r"""Initialize Demography.

**Arguments:**

- `ne`: Nonempty sequence of positive integer diploid effective population
  sizes, ordered oldest to newest. Each size represents `2 * ne` gene copies.
- `generations`: Positive integer durations between successive sizes, with
  exactly one fewer entry than `ne`.

Lists and concrete one-dimensional integer arrays are converted to host
tuples. Construct outside JIT.

**Returns:**

`None`.

**Raises:**

`ValueError` for invalid lengths, noninteger entries, or nonpositive entries.
Population sizes and durations are structural, not differentiable parameters.
"""


class IntervalDiagnostics(NamedTuple):
    r"""Numerical outcome for one complete demographic interval.

    `source_ne` and `destination_ne` are diploid population sizes; `generations`
    includes both the hold and final boundary generation. `error_estimate`
    covers only the hold power approximation, while `converged` and `status`
    also account for nonfinite boundary output.

    **Fields:**

    - `source_ne`, `destination_ne`: Python integers giving diploid sizes.
    - `generations`: Interval duration, including its final boundary generation.
    - `error_estimate`: Scalar absolute L1 hold-power estimate, zero without a hold.
    - `converged`: Scalar boolean for the complete interval.
    - `status`: First hold or boundary failure, or successful `EvolutionStatus`.
    """

    source_ne: int
    destination_ne: int
    generations: int
    error_estimate: jax.Array
    converged: jax.Array
    status: EvolutionStatus


# Document the generated constructor without replacing its implementation.
IntervalDiagnostics.__new__.__module__ = __name__
IntervalDiagnostics.__new__.__doc__ = r"""Construct an immutable result without validation.

**Arguments:**

The fields documented on `IntervalDiagnostics`, supplied by position or keyword.

**Returns:**

An `IntervalDiagnostics` holding the supplied values. Normal evolution calls construct this
result automatically; constructing it directly performs no numerical checks.
"""


class DemographicResult(NamedTuple):
    r"""Final mass and per-interval numerical outcomes, without a total error certificate.

    `status` retains the first numerical failure. Identity histories have empty
    `diagnostics` and still check mass finiteness. No factorization or basis is retained.

    **Fields:**

    - `distribution`: Float64 mass of length `2 * final_ne + 1`, or
      `2 * final_ne - 1` for segregating evolution; not normalized.
    - `converged`: Scalar boolean, true when the entire history succeeded.
    - `status`: First numerical failure in chronological order, or success.
    - `diagnostics`: Tuple of `IntervalDiagnostics`, one per interval in order.
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

The fields documented on `DemographicResult`, supplied by position or keyword.

**Returns:**

A `DemographicResult` holding the supplied values. Normal evolution calls construct this
result automatically; constructing it directly performs no numerical checks.
"""


@final
class DemographicTransition(eqx.Module):
    r"""Compose constant-size evolution and rectangular boundary transitions.

    An interval of g generations holds its source size for g-1 generations,
    then changes size on its last generation. Sequential provides
    a matched expectation transpose. Both Krylov methods support all
    backends for forward evolution only. Frequency-map parameters remain differentiable array leaves.
    """

    demography: Demography
    frequency_map: AbstractFrequencyMap
    method: AbstractEpochMethod
    _sizes: tuple[int, ...]
    _backends: tuple[AbstractTransitionBackend, ...]
    _boundary_backends: tuple[AbstractTransitionBackend, ...]

    def __init__(
        self,
        demography: Demography,
        *,
        backend_factory: Callable[[int], AbstractTransitionBackend],
        frequency_map: AbstractFrequencyMap,
        boundary_backend_factory: Callable[[int], AbstractTransitionBackend]
        | None = None,
        method: AbstractEpochMethod | None = None,
    ) -> None:
        r"""Prepare reusable geometry outside JIT.

        **Arguments:**

        - `demography`: Population history, oldest to newest.
        - `backend_factory`: Factory receiving a gene-copy count for constant-size geometry.
        - `frequency_map`: Process shared by every generation.
        - `boundary_backend_factory`: Factory for rectangular transitions; defaults to the
          hold geometry for all backends.
        - `method`: Geometry-free epoch algorithm; defaults to `Sequential()`.

        **Returns:**

        `None`. Prepares one geometry per distinct size, sharing hold and boundary
        geometry when the same factory is used.

        **Raises:**

        `TypeError` for unsupported combinations, including histories without holds;
        `ValueError` for incorrect backend sizes. RationalKrylov selects dense LU
        for Jacobi and GMRES for other transition backends. Dense is intended for
        small reference calculations; optimized backends avoid a dense population
        matrix.
        """
        if not isinstance(demography, Demography):
            raise TypeError("demography must be a Demography")
        if not isinstance(frequency_map, AbstractFrequencyMap):
            raise TypeError(
                "frequency_map must be an AbstractFrequencyMap; wrap functions in FrequencyMap"
            )
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable")
        if boundary_backend_factory is not None and not callable(
            boundary_backend_factory
        ):
            raise TypeError("boundary_backend_factory must be callable")
        self.demography = demography
        self.frequency_map = frequency_map
        self.method = Sequential() if method is None else method
        if not isinstance(self.method, AbstractEpochMethod):
            raise TypeError("method must be an AbstractEpochMethod")
        self._sizes = tuple(dict.fromkeys(demography.ne))
        self._backends = tuple(backend_factory(2 * size) for size in self._sizes)
        for size, geometry in zip(self._sizes, self._backends, strict=True):
            if not isinstance(geometry, AbstractTransitionBackend):
                raise TypeError(
                    "backend_factory must return an AbstractTransitionBackend"
                )
            if geometry.gene_copies != 2 * size:
                raise ValueError(
                    "backend_factory must preserve the requested gene-copy count"
                )
            self.method.validate(geometry, frequency_map)
        if (
            boundary_backend_factory is None
            or boundary_backend_factory is backend_factory
        ):
            self._boundary_backends = self._backends
        else:
            self._boundary_backends = tuple(
                boundary_backend_factory(2 * size) for size in self._sizes
            )

    def __check_init__(self) -> None:
        """Require rectangular transition geometry at all boundaries."""
        for size, backend in zip(self._sizes, self._boundary_backends, strict=True):
            if not isinstance(backend, AbstractTransitionBackend):
                raise TypeError(
                    "boundary_backend_factory must return an AbstractTransitionBackend"
                )
            if backend.gene_copies != 2 * size:
                raise ValueError(
                    "boundary_backend_factory must preserve the requested gene-copy count"
                )

    def _transition(self, source_ne: int, destination_ne: int) -> Transition:
        return Transition(
            self._boundary_backends[self._sizes.index(destination_ne)],
            self.frequency_map,
            source_gene_copies=2 * source_ne,
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

        **Arguments:**

        - `distribution`: Real mass of length `2 * initial_ne + 1`, or
          `2 * initial_ne - 1` when segregating. Converted to float64.
        - `segregating`: Discard endpoint mass after each generation.
        - `singleton_mutation_rate`: Finite nonnegative scalar per-copy rate,
          allowed only when segregating; defaults to zero. Adds
          `2 * destination_ne * singleton_mutation_rate` at count one after
          every generation. Recurrent mutation belongs in the frequency map.
        - `throw`: Raise on numerical failure; under vmap may fail the whole batch.

        **Returns:**

        `DemographicResult` containing final mass and each interval's outcome.
        Output length is `2 * final_ne + 1`, or `2 * final_ne - 1` when segregating.
        No mass normalization occurs.

        **Raises:**

        `ValueError` for invalid shapes, flags, or singleton input in full-state
        mode. Invalid rates or process domains raise even with `throw=False`.
        Other numerical failures raise with `throw=True` and return status with false.
        """
        if not isinstance(segregating, bool):
            raise ValueError("segregating must be a Python bool")
        if not segregating and singleton_mutation_rate is not None:
            raise ValueError("singleton_mutation_rate requires segregating=True")
        rate = None
        if segregating:
            rate = jnp.asarray(
                0.0 if singleton_mutation_rate is None else singleton_mutation_rate,
                dtype=jnp.float64,
            )
            if rate.shape != ():
                raise ValueError("singleton_mutation_rate must be scalar")
            distribution = eqx.error_if(
                jnp.asarray(distribution, dtype=jnp.float64),
                ~jnp.isfinite(rate) | (rate < 0),
                "singleton_mutation_rate must be finite and nonnegative",
            )
        return self._advance(
            distribution, segregating=segregating, singleton_rate=rate, throw=throw
        )

    def _advance(self, distribution, *, segregating, singleton_rate, throw):
        distribution = jnp.asarray(distribution, dtype=jnp.float64)
        size = 2 * self.demography.initial_ne + (-1 if segregating else 1)
        if distribution.shape != (size,):
            raise ValueError(f"distribution must have {size} entries")
        # Validate every prepared domain even for identity/boundary-only histories.
        for backend in self._backends:
            distribution = validate_domain(
                backend,
                distribution,
                self.frequency_map,
                reduced=self.method.uses_reduced_representation,
            )
        initial_result = _epoch_result(distribution, jnp.asarray(0.0), 0.0, False)
        status = initial_result.status
        diagnostics = []
        for source, destination, duration in zip(
            self.demography.ne[:-1],
            self.demography.ne[1:],
            self.demography.generations,
            strict=True,
        ):
            # The final generation belongs to the rectangular boundary action.
            # Holding for duration instead would add an extra generation and,
            # in segregating mode, an extra immigration event at the source size.
            error = jnp.asarray(0.0)
            interval_status = EvolutionStatus.successful
            if duration > 1:
                immigration = (
                    jnp.zeros_like(distribution).at[0].set(2 * source * singleton_rate)
                    if segregating
                    else None
                )
                problem = prepare_epoch(
                    self.method,
                    self._backends[self._sizes.index(source)],
                    distribution,
                    frequency_map=self.frequency_map,
                    generations=duration - 1,
                    segregating=segregating,
                    immigration=immigration,
                )
                result = self.method.solve(problem, throw=False)
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
    def expect(self, values: jax.Array) -> jax.Array:
        r"""Compute the matched expectation transpose for Sequential histories.

        **Arguments:**

        - `values`: Real vector of length `2 * final_ne + 1`, converted to float64.

        **Returns:**

        Float64 expectations of length `2 * initial_ne + 1`.

        **Raises:**

        `ValueError` for incompatible shapes or a non-Sequential method, which
        does not provide a matched transpose of its input-dependent approximation.
        """
        if not isinstance(self.method, Sequential):
            raise ValueError(
                "expect requires Sequential; adaptive epoch actions have no matched transpose"
            )
        values = jnp.asarray(values, dtype=jnp.float64)
        if values.shape != (2 * self.demography.final_ne + 1,):
            raise ValueError("values must have 2 * final_ne + 1 entries")
        # Transposition reverses composition: apply the last boundary's
        # expectation first, then its source-size hold, moving backward in time.
        for source, destination, duration in reversed(
            tuple(
                zip(
                    self.demography.ne[:-1],
                    self.demography.ne[1:],
                    self.demography.generations,
                    strict=True,
                )
            )
        ):
            values = self._transition(source, destination).expect(values)
            if duration > 1:
                backend = self._backends[self._sizes.index(source)]
                assert isinstance(backend, AbstractTransitionBackend)
                transition = Transition(backend, self.frequency_map)
                values = jax.lax.fori_loop(
                    0,
                    duration - 1,
                    lambda _, current, transition=transition: transition.expect(
                        current
                    ),
                    values,
                )
        return values
