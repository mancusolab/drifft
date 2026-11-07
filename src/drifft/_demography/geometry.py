"""Demographic schedules and reusable geometry for their holds and boundaries."""

# pattern: Functional Core
import operator
from collections.abc import Callable, Iterable, Iterator
from typing import final

import equinox as eqx
import jax

from .._backend import AbstractTransitionBackend


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
          size needed for a hold. Use `functools.partial` to supply backend settings.
        - `boundary_backend_factory`: Optional factory for rectangular boundary
          transitions, prepared only for destination sizes. Defaults to the hold
          factory, sharing geometry for sizes used in both roles.

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


@final
class DemographicGeometry(eqx.Module):
    """Prepared population-size backends and their immutable demographic schedule.

    Construct with [`Demography.prepare`][drifft.Demography.prepare] outside JIT. Geometry contains no bound
    frequency map or epoch method and can be shared across parameter evaluations.
    Backend approximation domains still constrain the processes used with it.

    `geometry[size]` returns the shared epoch backend for a diploid population
    size used by a constant-size hold (an interval longer than one generation).
    Iteration and `len(geometry)` expose only these hold sizes, excluding
    boundary-only sizes. An identity history prepares no backends. Use
    `geometry.demography` for the full chronological schedule.
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
        """Prepare only scheduled holds and destinations; prefer [`Demography.prepare`][drifft.Demography.prepare]."""
        if not isinstance(demography, Demography):
            raise TypeError("demography must be a Demography")
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable")
        if boundary_backend_factory is not None and not callable(
            boundary_backend_factory
        ):
            raise TypeError("boundary_backend_factory must be callable")
        self.demography = demography
        hold_sizes = dict.fromkeys(
            size
            for size, duration in zip(
                demography.pop_sizes[:-1], demography.generations, strict=True
            )
            if duration > 1
        )
        boundary_sizes = dict.fromkeys(demography.pop_sizes[1:])
        self._backends = {size: backend_factory(size) for size in hold_sizes}
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
            self._boundary_backends = {
                size: self._backends[size]
                if size in self._backends
                else backend_factory(size)
                for size in boundary_sizes
            }
        else:
            self._boundary_backends = {
                size: boundary_backend_factory(size) for size in boundary_sizes
            }

    def __getitem__(self, pop_size: int) -> AbstractTransitionBackend:
        """Return the prepared epoch backend for a concrete diploid population size.

        The key is a population size, not an interval index. Repeated sizes
        with holds share one backend. Boundary-only sizes are not returned, even
        when factories are shared. A size without a hold raises `KeyError`;
        lookup does not prepare geometry.
        """
        return self._backends[pop_size]

    def __iter__(self) -> Iterator[int]:
        """Iterate distinct hold population sizes, not chronological epochs.

        Use `demography.pop_sizes` and `demography.generations` for the ordered
        schedule, including repeated sizes. Iteration order here is unspecified.
        """
        return iter(self._backends)

    def __len__(self) -> int:
        """Return the number of distinct hold sizes, not the interval count."""
        return len(self._backends)

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
