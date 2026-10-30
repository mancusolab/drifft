"""Population histories and their demographic transitions."""

# pattern: Functional Core
import operator
from collections.abc import Callable, Iterable
from typing import NamedTuple, final

import equinox as eqx
import jax
import jax.numpy as jnp

from ._epochs import (
    AbstractEpochMethod,
    EvolutionStatus,
    Sequential,
    _epoch_result,
    _prepare_evolution,
)
from ._frequency import AbstractFrequencyMap
from ._operator import Transition
from .backends import AbstractTransitionBackend, Jacobi


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
    """

    source_ne: int
    destination_ne: int
    generations: int
    error_estimate: jax.Array
    converged: jax.Array
    status: EvolutionStatus


class DemographicResult(NamedTuple):
    r"""Final mass and per-interval numerical outcomes, without a total error certificate.

    `status` retains the first numerical failure. Identity histories have empty
    `diagnostics` and still check mass finiteness. No factorization or basis is retained.
    """

    distribution: jax.Array
    converged: jax.Array
    status: EvolutionStatus
    diagnostics: tuple[IntervalDiagnostics, ...]


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
        backend: Callable[[int], AbstractTransitionBackend],
        frequency_map: AbstractFrequencyMap,
        boundary_backend: Callable[[int], AbstractTransitionBackend] | None = None,
        method: AbstractEpochMethod | None = None,
    ) -> None:
        r"""Prepare reusable geometry outside JIT.

        **Arguments:**

        - `demography`: Population history, oldest to newest.
        - `backend`: Factory receiving a gene-copy count for constant-size geometry.
        - `frequency_map`: Process shared by every generation.
        - `boundary_backend`: Factory for rectangular transitions; defaults to the
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
        if not callable(backend):
            raise TypeError("backend must be callable")
        if boundary_backend is not None and not callable(boundary_backend):
            raise TypeError("boundary_backend must be callable")
        self.demography = demography
        self.frequency_map = frequency_map
        self.method = Sequential() if method is None else method
        if not isinstance(self.method, AbstractEpochMethod):
            raise TypeError("method must be an AbstractEpochMethod")
        self._sizes = tuple(dict.fromkeys(demography.ne))
        self._backends = tuple(backend(2 * size) for size in self._sizes)
        for size, geometry in zip(self._sizes, self._backends, strict=True):
            if not isinstance(geometry, AbstractTransitionBackend):
                raise TypeError("backend must return an AbstractTransitionBackend")
            if geometry.gene_copies != 2 * size:
                raise ValueError("backend must preserve the requested gene-copy count")
            self.method.validate(geometry, frequency_map)
        if boundary_backend is None or boundary_backend is backend:
            self._boundary_backends = self._backends
        else:
            self._boundary_backends = tuple(
                boundary_backend(2 * size) for size in self._sizes
            )

    def __check_init__(self) -> None:
        """Require rectangular transition geometry at all boundaries."""
        for size, backend in zip(self._sizes, self._boundary_backends, strict=True):
            if not isinstance(backend, AbstractTransitionBackend):
                raise TypeError(
                    "boundary_backend must return an AbstractTransitionBackend"
                )
            if backend.gene_copies != 2 * size:
                raise ValueError(
                    "boundary_backend must preserve the requested gene-copy count"
                )

    def _transition(self, source_ne: int, destination_ne: int) -> Transition:
        return Transition(
            self._boundary_backends[self._sizes.index(destination_ne)],
            self.frequency_map,
            source_gene_copies=2 * source_ne,
        )

    @eqx.filter_jit
    def advance(
        self, distribution: jax.Array, *, throw: bool = True
    ) -> DemographicResult:
        r"""Propagate full population mass to the final generation boundary.

        **Arguments:**

        - `distribution`: Real vector of length `2 * initial_ne + 1`, converted to float64.
        - `throw`: Raise on numerical failure; False returns status and diagnostics.
          Under vmap a failure with True may fail the whole batch.

        **Returns:**

        `DemographicResult` containing final mass and each interval's outcome.

        **Raises:**

        `ValueError` for incompatible shapes; runtime errors for invalid process
        domains or, with `throw=True`, numerical failure. No mass normalization occurs.
        """
        return self._advance(
            distribution, segregating=False, mutation_rate=None, throw=throw
        )

    @eqx.filter_jit
    def advance_segregating(
        self,
        distribution: jax.Array,
        *,
        mutation_rate: float | jax.Array = 0.0,
        throw: bool = True,
    ) -> DemographicResult:
        r"""Evolve interior mass with singleton immigration after every generation.

        **Arguments:**

        - `distribution`: Real interior mass of length `2 * initial_ne - 1`.
        - `mutation_rate`: Finite nonnegative scalar per-copy rate. Each generation
          adds `2 * destination_ne * mutation_rate` at count one.
        - `throw`: Raise on numerical failure; under vmap may fail the whole batch.

        **Returns:**

        `DemographicResult`; the distribution has length `2 * final_ne - 1`.

        **Raises:**

        `ValueError` for invalid shapes. Invalid mutation rates and process domains
        raise even with `throw=False`; numerical failures otherwise return status.
        """
        rate = jnp.asarray(mutation_rate, dtype=jnp.float64)
        if rate.shape != ():
            raise ValueError("mutation_rate must be scalar")
        distribution = eqx.error_if(
            jnp.asarray(distribution, dtype=jnp.float64),
            ~jnp.isfinite(rate) | (rate < 0),
            "mutation_rate must be finite and nonnegative",
        )
        return self._advance(
            distribution, segregating=True, mutation_rate=rate, throw=throw
        )

    def _advance(self, distribution, *, segregating, mutation_rate, throw):
        distribution = jnp.asarray(distribution, dtype=jnp.float64)
        size = 2 * self.demography.initial_ne + (-1 if segregating else 1)
        if distribution.shape != (size,):
            raise ValueError(f"distribution must have {size} entries")
        # Validate every prepared Jacobi domain even for identity/boundary-only histories.
        for backend in self._backends:
            if isinstance(backend, Jacobi) and not isinstance(self.method, Sequential):
                distribution, _ = backend._validate_mass(
                    distribution, self.frequency_map
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
            error = jnp.asarray(0.0)
            interval_status = EvolutionStatus.successful
            if duration > 1:
                immigration = (
                    (jnp.zeros_like(distribution).at[0].set(2 * source * mutation_rate))
                    if segregating
                    else None
                )
                problem = _prepare_evolution(
                    self.method,
                    self._backends[self._sizes.index(source)],
                    distribution,
                    self.frequency_map,
                    duration - 1,
                    segregating,
                    immigration,
                )
                result = self.method.solve(problem, throw=False)
                distribution, error, interval_status = (
                    result.distribution,
                    result.error_estimate,
                    result.status,
                )
            transition = self._transition(source, destination)
            if segregating:
                distribution = transition.advance_segregating(distribution)
                distribution = distribution.at[0].add(2 * destination * mutation_rate)
            else:
                distribution = transition.advance(distribution)
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
