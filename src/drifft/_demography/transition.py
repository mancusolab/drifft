"""Public demographic transition API and input validation.

Geometry is prepared separately; history actions and their derivative rules
are supplied by actions.py and autodiff.py respectively.
"""

# pattern: Functional Core
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp

from .._evolution.methods import AbstractEpochMethod, Sequential
from .._frequency import AbstractFrequencyMap
from .actions import DemographicExpectation, DemographicResult
from .autodiff import _demographic_advance, _demographic_expect
from .geometry import DemographicGeometry, Demography


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
        method/backend/process combinations for the scheduled holds.
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
        for size in geometry:
            self.method.validate(geometry[size], frequency_map)

    @property
    def demography(self) -> Demography:
        """The population schedule associated with the prepared geometry."""
        return self.geometry.demography

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
        with false. Sensitivity inner-solve failures and nonfinite outputs raise
        even with `throw=False`; additional Krylov sensitivity-error checks are
        opt-in via the method's `check_sensitivity=True` setting. Higher-order
        Krylov operator sensitivities raise explicitly.
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
        configuration. Numerical convergence and sensitivity numerical failures
        raise at runtime; extra sensitivity-error checks apply only when the
        method sets `check_sensitivity=True`. Higher-order Krylov operator
        sensitivities raise explicitly.
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
