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


@final
class DemographicTransition(eqx.Module):
    r"""Prepared evolution through a demography under a shared frequency map.

    Let $P_{a,b}$ map population size $a$ to $b$. An interval of $g$ generations
    has expectation operator $A=P_{a,a}^{g-1}P_{a,b}$: hold the source size, then
    change size on the final generation. Across intervals, `expect` applies the
    product of these operators; `advance` applies its transpose to initial mass.
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
        r"""Prepare the demographic transition outside JIT.

        **Arguments:**

        - `demography`: Validated population history.
        - `backend`: Factory from a gene-copy count to an `AbstractTransitionBackend`.
        - `frequency_map`: `AbstractFrequencyMap` shared by all generations.

        **Returns:**

        `None`. Prepares one backend per distinct population size.

        **Raises:**

        `TypeError` for invalid argument or factory-result types; `ValueError` for
        a backend with the wrong size. Backend construction errors propagate.
        """
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
        r"""Propagate mass to the final generation boundary.

        **Arguments:**

        - `distribution`: Finite real vector of length `2 * demography.initial_ne + 1`,
          converted to float64. Signed entries are supported.

        **Returns:**

        A float64 vector of length `2 * demography.final_ne + 1`.

        **Raises:**

        `ValueError` for incompatible input or map-output shapes.
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
        r"""Compute terminal-value expectations at the initial boundary.

        **Arguments:**

        - `values`: Finite real vector of length `2 * demography.final_ne + 1`,
          converted to float64. Signed entries are supported.

        **Returns:**

        A float64 vector of length `2 * demography.initial_ne + 1`.

        **Raises:**

        `ValueError` for incompatible input or map-output shapes.
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
