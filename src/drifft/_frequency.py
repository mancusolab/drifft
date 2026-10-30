"""Composable selection and mutation maps on allele frequencies."""

# pattern: Functional Core
import abc
from collections.abc import Callable
from typing import final

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
from jax.experimental import hijax


def _as_float64(value: float | jax.Array) -> jax.Array:
    """Normalize parameter leaves and retain useful generated constructor types."""
    return jnp.asarray(value, dtype=jnp.float64)


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


def _diploid_selection_impl(
    s_het: jax.Array | float, s_hom: jax.Array | float, frequencies: jax.Array
) -> jax.Array:
    """Evaluate diploid survival without forming large squared fitnesses."""
    ancestral = (1 - frequencies) * ((1 - frequencies) + (1 - s_het) * frequencies)
    derived = frequencies * (
        (1 - s_het) * (1 - frequencies) + (1 - s_hom) * frequencies
    )
    return derived / (ancestral + derived)


def _diploid_selection_jvp(primals, tangents):
    """Factor analytic partials to avoid overflow and quotient cancellation."""
    s_het, s_hom, x = primals
    ds_het, ds_hom, dx = map(hijax.instantiate_zeros, tangents)
    u = 1 - x
    het, hom = 1 - s_het, 1 - s_hom
    ancestral = u * (u + het * x)
    derived = x * (het * u + hom * x)
    mean_fitness = ancestral + derived
    u_over_mean = u / mean_fitness
    hom_x_over_mean = hom * x / mean_fitness
    # dq/dx = (het*u**2 + 2*hom*x*u + het*hom*x**2) / mean_fitness**2.
    # Divide factors separately; neither the fitness products nor D**2 need
    # be representable when the derivative itself is finite.
    partial_x = (
        (het * u / mean_fitness) * u_over_mean
        + 2 * hom_x_over_mean * u_over_mean
        + (het * x / mean_fitness) * hom_x_over_mean
    )
    partial_het = (x * u / mean_fitness) * (hom_x_over_mean * x - u_over_mean * u)
    partial_hom = -(x * x / mean_fitness) * (ancestral / mean_fitness)
    return _diploid_selection(*primals), (
        partial_het * ds_het + partial_hom * ds_hom + partial_x * dx
    )


class _DiploidSelection(hijax.HiPrim):
    """Selection with stable analytic derivatives in every AD mode."""

    def __init__(self, *avals):
        """Declare broadcast output shape and dtype from the primal formula."""
        self.in_avals = avals
        shape = jax.eval_shape(_diploid_selection_impl, *avals)
        self.out_aval = jax.core.ShapedArray(shape.shape, shape.dtype)
        self.params = {}
        super().__init__()

    def expand(self, *args):
        """Evaluate the selection probabilities."""
        return _diploid_selection_impl(*args)

    def jvp(self, primals, tangents):
        """Use factored analytic partials, including for higher derivatives."""
        return _diploid_selection_jvp(primals, tangents)

    lin, linearized = hijax.linearize_from_jvp
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

    def batch_dim_rule(self, axis_data, dims):
        """Map outputs only when a primal input is mapped."""
        return 0 if any(dim is not None for dim in dims) else None


def _diploid_selection(*args):
    """Bind the selection rule through HiJAX."""
    return _DiploidSelection(*(jax.typeof(arg) for arg in args))(*args)


@final
class DiploidSelection(AbstractFrequencyMap):
    r"""Diploid viability selection with independent genotype fitness reductions.

    Genotypes 0/0, 0/1, and 1/1 have fitnesses $1$, $1-s_{\mathrm{het}}$,
    and $1-s_{\mathrm{hom}}$. Under random mating, offspring allele frequency is

    $$q(x)=\frac{(1-s_{\mathrm{hom}})x^2+(1-s_{\mathrm{het}})x(1-x)}
    {(1-x)^2+2(1-s_{\mathrm{het}})x(1-x)+(1-s_{\mathrm{hom}})x^2}.$$

    Both coefficients are explicit; there is no default dominance relationship
    or clipping. Negative coefficients represent fitness advantages. Mutation
    is a separate map, applied in the order specified by `Composition`.
    """

    s_het: jax.Array = eqx.field(converter=_as_float64)
    s_hom: jax.Array = eqx.field(converter=_as_float64)

    def __check_init__(self) -> None:
        """Require scalar coefficients without inspecting traced values."""
        if self.s_het.ndim != 0 or self.s_hom.ndim != 0:
            raise ValueError("selection coefficients must be scalar")

    def apply(self, frequencies: jax.Array) -> jax.Array:
        r"""Return the fitness-weighted offspring allele frequencies.

        **Arguments:**

        - `frequencies`: Float64 array of allele-1 frequencies in `[0, 1]`.

        **Returns:**

        Selected frequencies with the same shape as the input.
        """
        return _diploid_selection(self.s_het, self.s_hom, frequencies)


# Document the generated constructor without replacing its implementation.
DiploidSelection.__init__.__doc__ = r"""Initialize DiploidSelection.

**Arguments:**

- `s_het`: Heterozygote fitness reduction, finite and at most one.
- `s_hom`: Derived-homozygote fitness reduction, finite and strictly below one.

Both parameters become dynamic float64 scalar leaves. Enable JAX x64 before
construction. The strict homozygote bound keeps mean fitness positive at
fixation; a lethal derived homozygote is outside this map's domain.

**Returns:**

`None`.

**Raises:**

`ValueError` for nonscalar coefficients.

!!! warning "Parameter domain"

    Value constraints are caller preconditions, including under JIT and vmap.
    Outputs and parameters are not clipped. All backends support this map.
    Jacobi's reduced Krylov assembly requires both coefficients to have absolute
    value at most its configured `max_strength`, which is strictly below one.
"""


@final
class Neutral(AbstractFrequencyMap):
    r"""Identity frequency map for neutral reproduction without mutation.

    This is `DiploidSelection(0, 0)`, simplified to $q(x)=x$ to preserve
    the input dtype and avoid unnecessary arithmetic.
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

    For allele-1 frequency $x$ and directional rates
    $\mu_{0\to1}$ (`forward_rate`) and $\mu_{1\to0}$ (`backward_rate`),

    $$q(x)=x(1-\mu_{1\to0})+(1-x)\mu_{0\to1}.$$
    """

    forward_rate: jax.Array = eqx.field(default=0.0, converter=_as_float64)
    backward_rate: jax.Array = eqx.field(default=0.0, converter=_as_float64)

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

    Genotypes 0/0, 0/1, and 1/1 have fitnesses $1$, $1-s$, and $1$.
    This is the constrained diploid rule `DiploidSelection(s_het=s, s_hom=0)`;
    both homozygotes have equal fitness. `AdditiveSelection(s)` instead
    assigns fitnesses $1$, $1-s/2$, and $1-s$.

    For allele frequency $x$ and heterozygote disadvantage $s$,

    $$q_s(x) = \frac{x-sx(1-x)}{1-2sx(1-x)}.$$

    Uses the shared diploid numerical function. The named class also selects
    the specialized underdominance assembly supported by the Jacobi backend.
    """

    strength: jax.Array = eqx.field(converter=_as_float64)

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
        return _diploid_selection(self.strength, 0.0, frequencies)


# Document the generated constructor without replacing its implementation.
Underdominance.__init__.__doc__ = r"""Initialize Underdominance.

**Arguments:**

- `strength`: Heterozygote fitness reduction in `[0, 1]`, converted to a
  float64 JAX leaf. Changing its value does not change model structure.

**Returns:**

`None`.

**Raises:**

`ValueError` for nonscalar strength.

!!! warning "Parameter domain"

    Strength must be finite and in `[0, 1]`. This value constraint is unchecked,
    including under JIT and vmap. Enable JAX x64 before construction.
"""


@final
class AdditiveSelection(AbstractFrequencyMap):
    r"""Additive diploid fitness effects.

    Genotype fitnesses are $1$, $1-s/2$, and $1-s$. The coefficient is the
    derived-homozygote fitness reduction, as in `DominanceSelection(s, h=0.5)`.

    Uses the `DiploidSelection` numerical rule with coefficients derived from
    the current parameter leaves on every call.
    """

    s: jax.Array = eqx.field(converter=_as_float64)

    def __check_init__(self) -> None:
        """Require scalar parameters without inspecting traced values."""
        if self.s.ndim != 0:
            raise ValueError("selection parameters must be scalar")

    def apply(self, frequencies: jax.Array) -> jax.Array:
        r"""Apply the constrained diploid selection rule.

        **Arguments:**

        - `frequencies`: Float64 array of allele-1 frequencies in `[0, 1]`.

        **Returns:**

        Selected frequencies with the same shape as the input.
        """
        return _diploid_selection(self.s / 2, self.s, frequencies)


# Document the generated constructor without replacing its implementation.
AdditiveSelection.__init__.__doc__ = r"""Initialize AdditiveSelection.

**Arguments:**

- `s`: Derived-homozygote fitness reduction, finite and strictly below `1`.
  The heterozygote reduction is `s / 2`; negative values represent advantages.

Parameters become dynamic float64 scalar leaves. Enable JAX x64 before
construction. Value constraints are caller preconditions, including under JIT
and vmap; no clipping is applied. FFT and NUFFT support this rule. Jacobi
checks that both derived selection coefficients have absolute value at most
its configured `max_strength`.

**Returns:**

`None`.

**Raises:**

`ValueError` for nonscalar parameters.
"""


@final
class DominanceSelection(AbstractFrequencyMap):
    r"""Diploid selection with fitnesses $1$, $1-hs$, and $1-s$.

    Uses the `DiploidSelection` numerical rule with coefficients derived from
    the current parameter leaves on every call.
    """

    s: jax.Array = eqx.field(converter=_as_float64)

    h: jax.Array = eqx.field(converter=_as_float64)

    def __check_init__(self) -> None:
        """Require scalar parameters without inspecting traced values."""
        if self.s.ndim != 0 or self.h.ndim != 0:
            raise ValueError("selection parameters must be scalar")

    def apply(self, frequencies: jax.Array) -> jax.Array:
        r"""Apply the constrained diploid selection rule.

        **Arguments:**

        - `frequencies`: Float64 array of allele-1 frequencies in `[0, 1]`.

        **Returns:**

        Selected frequencies with the same shape as the input.
        """
        return _diploid_selection(self.h * self.s, self.s, frequencies)


# Document the generated constructor without replacing its implementation.
DominanceSelection.__init__.__doc__ = r"""Initialize DominanceSelection.

**Arguments:**

- `s`: Derived-homozygote fitness reduction, finite and strictly below one.
- `h`: Finite dominance coefficient, with finite `h * s <= 1`.

Parameters become dynamic float64 scalar leaves. Enable JAX x64 before
construction. Value constraints are caller preconditions, including under JIT
and vmap; no clipping is applied. FFT and NUFFT support this rule. Jacobi
checks that both derived selection coefficients have absolute value at most
its configured `max_strength`.

**Returns:**

`None`.

**Raises:**

`ValueError` for nonscalar parameters.
"""


@final
class RecessiveSelection(AbstractFrequencyMap):
    r"""Completely recessive fitness effects with fitnesses $1$, $1$, and $1-s$.

    Uses the `DiploidSelection` numerical rule with coefficients derived from
    the current parameter leaves on every call.
    """

    s: jax.Array = eqx.field(converter=_as_float64)

    def __check_init__(self) -> None:
        """Require scalar parameters without inspecting traced values."""
        if self.s.ndim != 0:
            raise ValueError("selection parameters must be scalar")

    def apply(self, frequencies: jax.Array) -> jax.Array:
        r"""Apply the constrained diploid selection rule.

        **Arguments:**

        - `frequencies`: Float64 array of allele-1 frequencies in `[0, 1]`.

        **Returns:**

        Selected frequencies with the same shape as the input.
        """
        return _diploid_selection(0.0, self.s, frequencies)


# Document the generated constructor without replacing its implementation.
RecessiveSelection.__init__.__doc__ = r"""Initialize RecessiveSelection.

**Arguments:**

- `s`: Derived-homozygote fitness reduction, finite and strictly below one.

Parameters become dynamic float64 scalar leaves. Enable JAX x64 before
construction. Value constraints are caller preconditions, including under JIT
and vmap; no clipping is applied. FFT and NUFFT support this rule. Jacobi
checks that both derived selection coefficients have absolute value at most
its configured `max_strength`.

**Returns:**

`None`.

**Raises:**

`ValueError` for nonscalar parameters.
"""


@final
class DominantSelection(AbstractFrequencyMap):
    r"""Completely dominant fitness effects with fitnesses $1$, $1-s$, and $1-s$.

    Uses the `DiploidSelection` numerical rule with coefficients derived from
    the current parameter leaves on every call.
    """

    s: jax.Array = eqx.field(converter=_as_float64)

    def __check_init__(self) -> None:
        """Require scalar parameters without inspecting traced values."""
        if self.s.ndim != 0:
            raise ValueError("selection parameters must be scalar")

    def apply(self, frequencies: jax.Array) -> jax.Array:
        r"""Apply the constrained diploid selection rule.

        **Arguments:**

        - `frequencies`: Float64 array of allele-1 frequencies in `[0, 1]`.

        **Returns:**

        Selected frequencies with the same shape as the input.
        """
        return _diploid_selection(self.s, self.s, frequencies)


# Document the generated constructor without replacing its implementation.
DominantSelection.__init__.__doc__ = r"""Initialize DominantSelection.

**Arguments:**

- `s`: Fitness reduction for carriers, finite and strictly below one.

Parameters become dynamic float64 scalar leaves. Enable JAX x64 before
construction. Value constraints are caller preconditions, including under JIT
and vmap; no clipping is applied. FFT and NUFFT support this rule. Jacobi
checks that both derived selection coefficients have absolute value at most
its configured `max_strength`.

**Returns:**

`None`.

**Raises:**

`ValueError` for nonscalar parameters.
"""


@final
class Overdominance(AbstractFrequencyMap):
    r"""Symmetric heterozygote advantage with fitnesses $1$, $1+s$, and $1$.

    Uses the `DiploidSelection` numerical rule with coefficients derived from
    the current parameter leaves on every call.
    """

    strength: jax.Array = eqx.field(converter=_as_float64)

    def __check_init__(self) -> None:
        """Require scalar parameters without inspecting traced values."""
        if self.strength.ndim != 0:
            raise ValueError("selection parameters must be scalar")

    def apply(self, frequencies: jax.Array) -> jax.Array:
        r"""Apply the constrained diploid selection rule.

        **Arguments:**

        - `frequencies`: Float64 array of allele-1 frequencies in `[0, 1]`.

        **Returns:**

        Selected frequencies with the same shape as the input.
        """
        return _diploid_selection(-self.strength, 0.0, frequencies)


# Document the generated constructor without replacing its implementation.
Overdominance.__init__.__doc__ = r"""Initialize Overdominance.

**Arguments:**

- `strength`: Finite, nonnegative heterozygote fitness advantage.

Parameters become dynamic float64 scalar leaves. Enable JAX x64 before
construction. Value constraints are caller preconditions, including under JIT
and vmap; no clipping is applied. FFT and NUFFT support this rule. Jacobi
checks that both derived selection coefficients have absolute value at most
its configured `max_strength`.

**Returns:**

`None`.

**Raises:**

`ValueError` for nonscalar parameters.
"""


def _multiplicative_selection_impl(r: jax.Array, frequencies: jax.Array) -> jax.Array:
    """Evaluate multiplicative selection without squaring the fitness ratio."""
    derived = r * frequencies
    return derived / ((1 - frequencies) + derived)


def _multiplicative_selection_jvp(primals, tangents):
    """Retain finite partials even when the mean fitness squared overflows."""
    r, x = primals
    dr, dx = map(hijax.instantiate_zeros, tangents)
    mean_fitness = (1 - x) + r * x
    partial_r = (x / mean_fitness) * ((1 - x) / mean_fitness)
    # XLA otherwise combines the two divisions into r / mean_fitness**2.
    r_over_mean = jax.lax.optimization_barrier(r / mean_fitness)
    partial_x = r_over_mean / mean_fitness
    return _multiplicative_selection(*primals), partial_r * dr + partial_x * dx


class _MultiplicativeSelection(hijax.HiPrim):
    """Selection with stable analytic derivatives in every AD mode."""

    def __init__(self, *avals):
        """Declare broadcast output shape and dtype from the primal formula."""
        self.in_avals = avals
        shape = jax.eval_shape(_multiplicative_selection_impl, *avals)
        self.out_aval = jax.core.ShapedArray(shape.shape, shape.dtype)
        self.params = {}
        super().__init__()

    def expand(self, *args):
        """Evaluate the selection probabilities."""
        return _multiplicative_selection_impl(*args)

    def jvp(self, primals, tangents):
        """Use factored analytic partials, including for higher derivatives."""
        return _multiplicative_selection_jvp(primals, tangents)

    lin, linearized = hijax.linearize_from_jvp
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

    def batch_dim_rule(self, axis_data, dims):
        """Map outputs only when a primal input is mapped."""
        return 0 if any(dim is not None for dim in dims) else None


def _multiplicative_selection(*args):
    """Bind the selection rule through HiJAX."""
    return _MultiplicativeSelection(*(jax.typeof(arg) for arg in args))(*args)


@final
class MultiplicativeSelection(AbstractFrequencyMap):
    r"""Multiplicative fitness effects with fitnesses $1$, $\rho$, and $\rho^2$.

    The API parameter `r` is the fitness ratio $\rho$. This is the diploid rule
    with coefficients $1-\rho$ and $1-\rho^2$. Cancelling the common factor gives
    $q(x)=\rho x/(1-x+\rho x)$, avoiding cancellation in the coefficients when
    fitness is small and overflow from squaring large $\rho$.
    """

    r: jax.Array = eqx.field(converter=_as_float64)

    def __check_init__(self) -> None:
        """Require scalar parameters without inspecting traced values."""
        if self.r.ndim != 0:
            raise ValueError("selection parameters must be scalar")

    def apply(self, frequencies: jax.Array) -> jax.Array:
        r"""Apply the constrained diploid selection rule.

        **Arguments:**

        - `frequencies`: Float64 array of allele-1 frequencies in `[0, 1]`.

        **Returns:**

        Selected frequencies with the same shape as the input.
        """
        return _multiplicative_selection(self.r, frequencies)


# Document the generated constructor without replacing its implementation.
MultiplicativeSelection.__init__.__doc__ = r"""Initialize MultiplicativeSelection.

**Arguments:**

- `r`: Finite, strictly positive fitness multiplier per derived allele.

Parameters become dynamic float64 scalar leaves. Enable JAX x64 before
construction. Value constraints are caller preconditions, including under JIT
and vmap; no clipping is applied. FFT and NUFFT support this rule. Jacobi
checks that both derived selection coefficients have absolute value at most
its configured `max_strength`.

**Returns:**

`None`.

**Raises:**

`ValueError` for nonscalar parameters.
"""
