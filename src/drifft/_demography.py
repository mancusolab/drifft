"""Population histories and their matrix-free demographic transitions."""

# pattern: Functional Core
import operator
from collections.abc import Callable, Iterable
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp

from ._frequency import AbstractFrequencyMap
from ._operator import Transition
from .backends import AbstractTransitionBackend


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

    **Arguments:**

    - `ne`: Nonempty sequence of positive integer diploid effective population
      sizes, ordered oldest to newest. Each size represents `2 * ne` gene copies.
    - `generations`: Positive integer durations between successive sizes, with
      exactly one fewer entry than `ne`.

    **Returns:**

    An immutable history with host tuples. Lists and concrete one-dimensional
    integer arrays are accepted; construct outside JIT. For each interval, hold
    the source size for `generations[i] - 1` generations, then change size on the
    final generation. Repeated sizes are valid. A single size and no durations
    describe zero generations. No equilibrium initialization is implied.

    **Raises:**

    `ValueError` for invalid lengths, noninteger entries, or nonpositive entries.
    Population sizes and durations are structural, not differentiable parameters.
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
        """Return the initial diploid population size."""
        return self.ne[0]

    @property
    def final_ne(self) -> int:
        """Return the final diploid population size."""
        return self.ne[-1]

    @property
    def total_generations(self) -> int:
        """Return the total number of Wright–Fisher generations evaluated."""
        return sum(self.generations)


@final
class DemographicTransition(eqx.Module):
    r"""Prepared evolution through a demography under a shared frequency map.

    **Arguments:**

    - `demography`: Population history with size changes at interval endpoints.
    - `backend`: Callable taking a gene-copy count and returning prepared geometry,
      such as `FFT`, `NUFFT`, or `functools.partial(NUFFT, accuracy=1e-11)`.
    - `frequency_map`: An `AbstractFrequencyMap` used at every generation.
      Its Equinox array parameters remain differentiable. Wrap custom functions
      in `FrequencyMap`, and combine processes with `Composition`.

    **Returns:**

    An immutable PyTree. Construct outside JIT: one backend is built per distinct
    population size. Geometry is independent of the frequency map. Public numerical
    methods compile automatically, and fixed-length loops support reverse-mode
    differentiation through its parameters. Changing the schedule may recompile.

    **Raises:**

    `TypeError` for invalid history, frequency map, or backend factory results;
    `ValueError` if a factory returns geometry with the wrong gene-copy count.
    Backend construction can also raise its documented configuration errors.

    **Failure Modes:**

    Enable JAX x64. The map must return one finite probability in `[0, 1]` per
    source frequency. No clipping, renormalization, or equilibrium solve occurs.
    Backend approximation errors can accumulate over generations; derivative
    accuracy retains the chosen backend's limitations. Reverse-mode memory can
    grow with history length. Only the final distribution is returned.
    """

    demography: Demography
    frequency_map: AbstractFrequencyMap
    _sizes: tuple[int, ...]
    _backends: tuple[AbstractTransitionBackend, ...]

    def __init__(
        self,
        demography: Demography,
        *,
        backend: Callable[[int], AbstractTransitionBackend],
        frequency_map: AbstractFrequencyMap,
    ) -> None:
        """Prepare each distinct population geometry once, outside transformations."""
        if not isinstance(demography, Demography):
            raise TypeError("demography must be a Demography")
        if not isinstance(frequency_map, AbstractFrequencyMap):
            raise TypeError(
                "frequency_map must be an AbstractFrequencyMap; wrap functions in FrequencyMap"
            )
        if not callable(backend):
            raise TypeError("backend must be callable")
        self.demography = demography
        self.frequency_map = frequency_map
        self._sizes = tuple(dict.fromkeys(demography.ne))
        self._backends = tuple(backend(2 * size) for size in self._sizes)

    def __check_init__(self) -> None:
        """Require valid destination geometry from the backend factory."""
        for size, backend in zip(self._sizes, self._backends, strict=True):
            if not isinstance(backend, AbstractTransitionBackend):
                raise TypeError("backend must return an AbstractTransitionBackend")
            if backend.gene_copies != 2 * size:
                raise ValueError("backend must preserve the requested gene-copy count")

    def _transition(self, source_ne: int, destination_ne: int) -> Transition:
        """Bind the current frequency map to existing geometry without caching probabilities."""
        return Transition(
            self._backends[self._sizes.index(destination_ne)],
            self.frequency_map,
            source_gene_copies=2 * source_ne,
        )

    @eqx.filter_jit
    def advance(self, distribution: jax.Array) -> jax.Array:
        r"""Propagate initial mass to the final generation boundary.

        **Arguments:**

        - `distribution`: Vector of length `2 * demography.initial_ne + 1`.

        **Returns:**

        Float64 vector of length `2 * demography.final_ne + 1`. Signed mass is
        supported; the input is neither mutated nor normalized.

        **Raises:**

        `ValueError` if the input does not match the initial population grid.
        """
        distribution = jnp.asarray(distribution, dtype=jnp.float64)
        if distribution.shape != (2 * self.demography.initial_ne + 1,):
            raise ValueError("distribution must have 2 * initial_ne + 1 entries")
        # Shapes vary between intervals, so only the fixed-size portions scan.
        for source, destination, duration in zip(
            self.demography.ne[:-1],
            self.demography.ne[1:],
            self.demography.generations,
            strict=True,
        ):
            if duration > 1:
                transition = self._transition(source, source)
                distribution = jax.lax.fori_loop(
                    0,
                    duration - 1,
                    lambda _, mass, transition=transition: transition.advance(mass),
                    distribution,
                )
            distribution = self._transition(source, destination).advance(distribution)
        return distribution

    @eqx.filter_jit
    def expect(self, values: jax.Array) -> jax.Array:
        r"""Evaluate terminal values conditional on each initial allele count.

        **Arguments:**

        - `values`: Vector of length `2 * demography.final_ne + 1`.

        **Returns:**

        Float64 expectations of length `2 * demography.initial_ne + 1`.
        This is the adjoint of `advance`, not evolution backward in time.

        **Raises:**

        `ValueError` if the input does not match the final population grid.
        """
        values = jnp.asarray(values, dtype=jnp.float64)
        if values.shape != (2 * self.demography.final_ne + 1,):
            raise ValueError("values must have 2 * final_ne + 1 entries")
        # Apply adjoints in reverse order: size change first, then the hold.
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
                transition = self._transition(source, source)
                values = jax.lax.fori_loop(
                    0,
                    duration - 1,
                    lambda _, current, transition=transition: transition.expect(
                        current
                    ),
                    values,
                )
        return values
