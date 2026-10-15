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

    A map takes allele frequencies $x$ to offspring probabilities $q(x)$,
    preserving the input shape. Concrete maps implement `apply`; `__call__`
    provides shorthand. Shared input assumptions belong to the map's class
    contract rather than being repeated on each method.

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
        r"""Transform allele frequencies into offspring probabilities.

        **Arguments:**

        - `frequencies`: Float64 array of allele frequencies in `[0, 1]`.

        **Returns:**

        An array of probabilities with the same shape as `frequencies`.
        """
        raise NotImplementedError

    def __call__(self, frequencies: jax.Array) -> jax.Array:
        r"""Call `apply` using function syntax.

        **Arguments:**

        - `frequencies`: Float64 array of allele frequencies in `[0, 1]`.

        **Returns:**

        The result of `self.apply(frequencies)`.
        """
        return self.apply(frequencies)


@final
class FrequencyMap(AbstractFrequencyMap):
    r"""Wrap a custom JAX-compatible function in the frequency-map interface.

    The wrapped function must be pure, preserve shape, and return finite
    probabilities in `[0, 1]`. Its parameters stay connected to differentiation;
    its output is returned without conversion, clipping, or normalization.

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
        r"""Apply the wrapped function.

        **Arguments:**

        - `frequencies`: Float64 array of allele frequencies in `[0, 1]`.

        **Returns:**

        The function's output, with the same shape as `frequencies`.
        """
        return self.function(frequencies)


@final
class Neutral(AbstractFrequencyMap):
    r"""Identity frequency map for neutral reproduction without mutation.

    The offspring probability equals the source frequency: $q(x)=x$.

    **Arguments:**

    None.

    **Returns:**

    A parameter-free Equinox map whose `apply` returns its input unchanged.

    **Failure Modes:**

    Frequency ranges and finiteness are caller preconditions. No runtime checks
    or random sampling are performed.
    """

    def apply(self, frequencies: jax.Array) -> jax.Array:
        r"""Return frequencies unchanged.

        **Arguments:**

        - `frequencies`: Float64 array of allele frequencies in `[0, 1]`.

        **Returns:**

        The input array, with its shape and dtype preserved.
        """
        return frequencies


@final
class Mutation(AbstractFrequencyMap):
    r"""Apply forward and back mutation to an allele frequency.

    For allele-1 frequency $p$, forward rate $u$, and backward rate $v$,

    $$q(p) = p(1-v) + (1-p)u.$$

    **Arguments:**

    - `forward_rate`: Per-generation probability of mutation from allele 0 to 1.
    - `backward_rate`: Per-generation probability of mutation from allele 1 to 0.

    Both rates default to zero and become dynamic float64 scalar leaves.

    **Returns:**

    An immutable map with differentiable rates. Each rate lies in `[0, 1]`;
    their sum need not be less than one. Use `Composition` to specify mutation
    before or after selection.

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
        r"""Apply forward and back mutation.

        **Arguments:**

        - `frequencies`: Float64 array of allele frequencies in `[0, 1]`.

        **Returns:**

        Mutated frequencies with the same shape as the input.
        """
        return (
            frequencies * (1 - self.backward_rate)
            + (1 - frequencies) * self.forward_rate
        )


@final
class Composition(AbstractFrequencyMap):
    r"""Apply frequency maps in the supplied left-to-right order.

    For component maps $q_1,\ldots,q_k$, the combined map is

    $$q(x)=q_k(\cdots q_2(q_1(x))\cdots).$$

    Selection and mutation generally do not commute, so their order is part of
    the model. Each component must preserve shape and return finite probabilities
    in `[0, 1]`; component failures propagate without additional validation.

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
        r"""Bind the ordered frequency maps.

        **Arguments:**

        - `*maps`: One or more `AbstractFrequencyMap` instances, applied left to right.

        **Returns:**

        `None`.

        **Raises:**

        `ValueError` for no maps; `TypeError` for a non-map component.
        """
        self.maps = maps

    def __check_init__(self) -> None:
        """Require a nonempty sequence of frequency-map components."""
        if not self.maps:
            raise ValueError("Composition requires at least one map; use Neutral()")
        if any(not isinstance(model, AbstractFrequencyMap) for model in self.maps):
            raise TypeError("components must be AbstractFrequencyMap instances")

    def apply(self, frequencies: jax.Array) -> jax.Array:
        r"""Apply the component maps from left to right.

        **Arguments:**

        - `frequencies`: Float64 array of allele frequencies in `[0, 1]`.

        **Returns:**

        The final component's output, with the same shape as the input.
        """
        for model in self.maps:
            frequencies = model.apply(frequencies)
        return frequencies


@final
class Underdominance(AbstractFrequencyMap):
    r"""Symmetric heterozygote disadvantage, callable on allele frequencies.

    For allele frequency $x$ and heterozygote disadvantage $s$,

    $$q_s(x) = \frac{x-sx(1-x)}{1-2sx(1-x)}.$$

    **Arguments:**

    - `strength`: Scalar selection coefficient in `[0, 1]`, converted to a
      float64 JAX leaf. Changing its value does not change model structure.

    **Returns:**

    An immutable frequency map whose strength remains differentiable. Both
    `apply(frequencies)` and callable shorthand evaluate the same selection rule.

    **Failure Modes:**

    Frequencies and strength must be finite and lie in `[0, 1]`; these are
    unchecked caller preconditions, including under JIT and vmap.
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
        r"""Apply heterozygote disadvantage.

        **Arguments:**

        - `frequencies`: Float64 array of allele frequencies in `[0, 1]`.

        **Returns:**

        Selected frequencies with the same shape as the input.
        """
        disadvantage = self.strength * frequencies * (1 - frequencies)
        return (frequencies - disadvantage) / (1 - 2 * disadvantage)
