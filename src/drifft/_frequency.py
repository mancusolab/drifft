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
    provides shorthand. Implementations must be pure and JAX-compatible;
    differentiable parameters are stored as array leaves.
    """

    @abc.abstractmethod
    def apply(self, frequencies: jax.Array) -> jax.Array:
        r"""Transform allele frequencies into offspring probabilities.

        **Arguments:**

        - `frequencies`: Float64 array of allele frequencies in `[0, 1]`.

        **Returns:**

        An array of finite probabilities in `[0, 1]` with the same shape as
        `frequencies`.

        !!! note "Input domain"

            Finiteness and the `[0, 1]` range are caller preconditions. Maps do
            not clip or normalize their outputs.
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


# Document the generated constructor without replacing its implementation.
FrequencyMap.__init__.__doc__ = r"""Initialize FrequencyMap.

**Arguments:**

- `function`: Callable returning one probability per input frequency. An
  Equinox callable can carry dynamic parameters. For closures, construct the
  function inside the differentiated calculation to retain parameter links.

**Returns:**

`None`.

**Raises:**

`TypeError` for non-callable functions. Output shape and probability ranges
follow `AbstractFrequencyMap`'s contract; transitions check output shape.
"""


@final
class Neutral(AbstractFrequencyMap):
    r"""Identity frequency map for neutral reproduction without mutation.

    The offspring probability equals the source frequency: $q(x)=x$.
    """

    def apply(self, frequencies: jax.Array) -> jax.Array:
        r"""Return frequencies unchanged.

        **Arguments:**

        - `frequencies`: Float64 array of allele frequencies in `[0, 1]`.

        **Returns:**

        The input array, with its shape and dtype preserved.
        """
        return frequencies


# Document the generated constructor without replacing its implementation.
Neutral.__init__.__doc__ = r"""Initialize Neutral.

**Arguments:**

None.

**Returns:**

`None`.

"""


@final
class Mutation(AbstractFrequencyMap):
    r"""Apply forward and back mutation to an allele frequency.

    For allele-1 frequency $p$, forward rate $u$, and backward rate $v$,

    $$q(p) = p(1-v) + (1-p)u.$$
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


# Document the generated constructor without replacing its implementation.
Mutation.__init__.__doc__ = r"""Initialize Mutation.

**Arguments:**

- `forward_rate`: Per-generation probability of mutation from allele 0 to 1.
- `backward_rate`: Per-generation probability of mutation from allele 1 to 0.

Both rates default to zero and become dynamic float64 scalar leaves.
Each rate lies in `[0, 1]`; their sum need not be less than one.

**Returns:**

`None`.

**Raises:**

`ValueError` for nonscalar rates.

!!! warning "Parameter domain"

    Rates must be finite and in `[0, 1]`. These value constraints are unchecked,
    including under JIT. Enable JAX x64 before construction.
"""


@final
class Composition(AbstractFrequencyMap):
    r"""Apply frequency maps in the supplied left-to-right order.

    For component maps $q_1,\ldots,q_k$, the combined map is

    $$q(x)=q_k(\cdots q_2(q_1(x))\cdots).$$

    Selection and mutation generally do not commute, so their order is part of
    the model. Each component must preserve shape and return finite probabilities
    in `[0, 1]`; component failures propagate without additional validation.
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


# Document the generated constructor without replacing its implementation.
Underdominance.__init__.__doc__ = r"""Initialize Underdominance.

**Arguments:**

- `strength`: Scalar selection coefficient in `[0, 1]`, converted to a
  float64 JAX leaf. Changing its value does not change model structure.

**Returns:**

`None`.

**Raises:**

`ValueError` for nonscalar strength.

!!! warning "Parameter domain"

    Strength must be finite and in `[0, 1]`. This value constraint is unchecked,
    including under JIT and vmap. Enable JAX x64 before construction.
"""
