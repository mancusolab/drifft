"""Composable selection and mutation maps on allele frequencies."""

# pattern: Functional Core
import abc
from collections.abc import Callable
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp


class AbstractFrequencyMap(eqx.Module):
    r"""Interface for transforming source frequencies into offspring probabilities.

    **Arguments:**

    Implement `apply(frequencies)` in concrete subclasses. Frequencies lie in
    `[0, 1]`; return a JAX array of the same shape with values in `[0, 1]`.

    **Returns:**

    A callable Equinox module. Store differentiable parameters as array leaves.
    Both `map.apply(frequencies)` and `map(frequencies)` evaluate the same rule.

    **Failure Modes:**

    Range and finiteness are caller preconditions, not runtime checks. Map
    implementations must be JAX-compatible and deterministic. Use float64.
    """

    @abc.abstractmethod
    def apply(self, frequencies: jax.Array) -> jax.Array:
        """Return offspring probabilities with the same shape as frequencies."""
        raise NotImplementedError

    def __call__(self, frequencies: jax.Array) -> jax.Array:
        """Evaluate this map as shorthand for apply."""
        return self.apply(frequencies)


@final
class FrequencyMap(AbstractFrequencyMap):
    r"""Wrap a custom JAX-compatible function in the frequency-map interface.

    **Arguments:**

    - `function`: Callable returning one probability per input frequency. An
      Equinox callable can carry dynamic parameters. For closures, construct the
      function inside the differentiated calculation to retain parameter links.

    **Returns:**

    A map whose `apply` delegates to the supplied function.

    **Raises:**

    `TypeError` for non-callable functions. Output shape and probability ranges
    follow `AbstractFrequencyMap`'s contract; transitions check output shape.
    """

    function: Callable[[jax.Array], jax.Array]

    def __check_init__(self) -> None:
        """Require an evaluable custom frequency transformation."""
        if not callable(self.function):
            raise TypeError("function must be callable")

    def apply(self, frequencies: jax.Array) -> jax.Array:
        """Apply the custom function without detaching dynamic parameters."""
        return self.function(frequencies)


@final
class Neutral(AbstractFrequencyMap):
    """Identity frequency map, representing neutral reproduction without mutation."""

    def apply(self, frequencies: jax.Array) -> jax.Array:
        """Return source frequencies unchanged."""
        return frequencies


@final
class Mutation(AbstractFrequencyMap):
    r"""Apply forward and back mutation to an allele frequency.

    **Arguments:**

    - `forward_rate`: Per-generation probability of mutation from allele 0 to 1.
    - `backward_rate`: Per-generation probability of mutation from allele 1 to 0.

    Both rates default to zero and become dynamic float64 scalar leaves.

    **Returns:**

    A map computing `p * (1 - backward_rate) + (1 - p) * forward_rate`.
    Each rate must lie in `[0, 1]`; their sum need not be less than one.
    Use `Composition` to specify mutation before or after selection.

    **Raises:**

    `ValueError` for nonscalar rates. Finite rates in `[0, 1]` and input
    frequencies in `[0, 1]` are caller preconditions, including under JIT.
    """

    forward_rate: jax.Array = eqx.field(
        default=0.0, converter=lambda x: jnp.asarray(x, dtype=jnp.float64)
    )
    backward_rate: jax.Array = eqx.field(
        default=0.0, converter=lambda x: jnp.asarray(x, dtype=jnp.float64)
    )

    def __check_init__(self) -> None:
        """Require scalar rates without inspecting traced values."""
        if self.forward_rate.ndim != 0 or self.backward_rate.ndim != 0:
            raise ValueError("mutation rates must be scalar")

    def apply(self, frequencies: jax.Array) -> jax.Array:
        """Return frequencies after mutation in both directions."""
        return (
            frequencies * (1 - self.backward_rate)
            + (1 - frequencies) * self.forward_rate
        )


@final
class Composition(AbstractFrequencyMap):
    r"""Apply frequency maps in the supplied left-to-right order.

    **Arguments:**

    - `*maps`: One or more `AbstractFrequencyMap` instances. For example,
      `Composition(Underdominance(0.1), Mutation(1e-8, 1e-8))` applies selection
      before mutation. Reverse the arguments to reverse the biological order.

    **Returns:**

    An immutable composition whose component parameters remain dynamic leaves.

    **Raises:**

    `ValueError` for an empty composition; use `Neutral()` for the identity map.
    `TypeError` for non-map components; wrap custom functions in `FrequencyMap`.
    """

    maps: tuple[AbstractFrequencyMap, ...]

    def __init__(self, *maps: AbstractFrequencyMap) -> None:
        """Store the ordered components without evaluating their probabilities."""
        self.maps = maps

    def __check_init__(self) -> None:
        """Require a nonempty sequence of frequency-map components."""
        if not self.maps:
            raise ValueError("Composition requires at least one map; use Neutral()")
        if any(not isinstance(model, AbstractFrequencyMap) for model in self.maps):
            raise TypeError("components must be AbstractFrequencyMap instances")

    def apply(self, frequencies: jax.Array) -> jax.Array:
        """Apply each component to the preceding component's output."""
        for model in self.maps:
            frequencies = model.apply(frequencies)
        return frequencies


@final
class Underdominance(AbstractFrequencyMap):
    r"""Symmetric heterozygote disadvantage, callable on allele frequencies.

    **Arguments:**

    - `strength`: Scalar selection coefficient in `[0, 1]`, converted to a
      float64 JAX leaf. Changing its value does not change model structure.

    **Failure Modes:**

    Frequencies and strength must lie in `[0, 1]`; these are caller preconditions.
    Non-scalar strength raises `ValueError`. Keep x64 enabled.
    """

    strength: jax.Array = eqx.field(
        converter=lambda x: jnp.asarray(x, dtype=jnp.float64)
    )

    def __check_init__(self) -> None:
        """Require scalar selection without inspecting traced values."""
        if self.strength.ndim != 0:
            raise ValueError("strength must be scalar")

    def apply(self, frequencies: jax.Array) -> jax.Array:
        """Return offspring probabilities with the same shape as frequencies."""
        disadvantage = self.strength * frequencies * (1 - frequencies)
        return (frequencies - disadvantage) / (1 - 2 * disadvantage)
