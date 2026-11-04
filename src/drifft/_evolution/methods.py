"""Geometry-free evolution methods and their common orchestration."""

# pattern: Functional Core
import math
from abc import abstractmethod
from typing import ClassVar, final

import equinox as eqx
import jax
import jax.numpy as jnp

from .._backend import AbstractTransitionBackend
from .._frequency import AbstractFrequencyMap
from .autodiff import _cached_power_action, _frechet_action, _power_action
from .preparation import _prepare_epoch, _prepare_transition, validate_representation
from .problem import (
    EpochProblem,
    EpochResult,
    _epoch_result,
    _PowerProblem,
    _working_problem,
)
from .projection import _projected_result, project


class AbstractEpochMethod(eqx.Module):
    r"""An algorithm for a prepared linear power action, independent of geometry.

    Concrete methods declare `allows_reduced_representation`: true allows backend
    reduction during preparation; false retains population coordinates. Direct
    calls and demographic histories use the same declaration. Implement
    `validate` and `solve`; the default `advance` prepares and solves an epoch.
    """

    allows_reduced_representation: eqx.AbstractClassVar[bool]
    # This is a preparation requirement, not a test of the concrete method type.
    # Custom population-coordinate methods must declare False.

    @abstractmethod
    def validate(
        self, backend: AbstractTransitionBackend, frequency_map: AbstractFrequencyMap
    ) -> None:
        r"""Reject unsupported representation and process combinations.

        **Arguments:**

        - `backend`: Prepared transition geometry.
        - `frequency_map`: Process applied in each generation.

        **Returns:**

        `None`. Checks types and supported combinations, without evaluating
        dynamic parameter values. Runtime domain checks occur during preparation.

        **Raises:**

        `TypeError` for unsupported backend or map types. Sequential accepts
        all transition backends; reduced Jacobi powers restrict map compositions.
        """
        ...

    @abstractmethod
    def solve(self, problem: EpochProblem, *, throw: bool = True) -> EpochResult:
        r"""Evaluate a power problem returned by `prepare_epoch`.

        **Arguments:**

        - `problem`: Read-only `EpochProblem` with the working operator, initial
          state, exponent, reconstruction and result checks. Use `prepare_epoch`
          to obtain it from population-space inputs; custom methods should use
          `problem.result` to finish the action.
        - `throw`: Raise on numerical failure when true; otherwise return status.

        **Returns:**

        An `EpochResult` in population coordinates. Its error estimate covers
        the power approximation, excluding backend and representation errors.

        **Raises:**

        Runtime errors for numerical failure with `throw=True`. A failed batch
        member may fail the entire call. Krylov sensitivity failures raise even
        with `throw=False`; higher-order operator sensitivities are unsupported.
        """
        ...

    @eqx.filter_jit
    def advance(
        self,
        backend: AbstractTransitionBackend,
        distribution: jax.Array,
        *,
        frequency_map: AbstractFrequencyMap,
        generations: int,
        segregating: bool = False,
        immigration: jax.Array | None = None,
        throw: bool = True,
    ) -> EpochResult:
        r"""Bind geometry and evolve a constant-size interval.

        **Arguments:**

        - `backend`: Prepared geometry compatible with this method.
        - `distribution`: Real mass, converted to float64. Length is
          `2 * backend.diploid_size + 1`, or `2 * backend.diploid_size - 1` if segregating.
          Signed entries are supported; no normalization is applied.
        - `frequency_map`: Process shared by every generation.
        - `generations`: Nonnegative Python integer, static under JIT.
        - `segregating`: Discard endpoint mass at each generation.
        - `immigration`: Mass added after every generation, matching the input
          shape; `None` means no source. In segregating mode it is added after
          endpoint mass is discarded.
        - `throw`: Raise on numerical failure; under vmap this may fail the whole batch.

        **Returns:**

        An `EpochResult` in population coordinates; zero generations preserves
        the input without immigration. Sequential reports zero power-approximation
        error; Krylov estimates exclude backend and representation errors.
        Check `converged` with `throw=False`.

        **Raises:**

        `TypeError` for unsupported combinations; `ValueError` for static input errors.
        Runtime domain checks remain active even with `throw=False`; Krylov
        sensitivity failures also raise independently of `throw`. Sequential
        supports higher derivatives; higher-order Krylov operator sensitivities
        are unsupported.

        !!! example "Example: Evolve with immigration"

            ```python
            import jax
            import jax.numpy as jnp
            import drifft

            jax.config.update("jax_enable_x64", True)
            backend = drifft.Jacobi(diploid_size=64)
            initial = jnp.zeros(127).at[0].set(1.0)
            source = jnp.zeros_like(initial).at[0].set(128 * 1e-8)
            result = drifft.RationalKrylov(dimension=40).advance(
                backend, initial, frequency_map=drifft.Neutral(),
                generations=250, segregating=True, immigration=source,
            )
            final = result.distribution
            ```

            The source adds singleton mass after every generation, including
            the last. For automatic per-copy mutation-rate scaling across
            changing sizes, see [new mutations through time](../examples/mutation_input.md).

        !!! note "Generation timing"

            This call uses the supplied backend for all 250 generations.
            `Demography([64, 64], [250])` instead uses 249 hold generations and
            one boundary generation, even with equal sizes. Different numerical
            representations of these steps can produce different approximations
            to the same biological recurrence.
        """
        problem = prepare_epoch(
            self,
            backend,
            distribution,
            frequency_map=frequency_map,
            generations=generations,
            segregating=segregating,
            immigration=immigration,
        )
        return self.solve(problem, throw=throw)


def prepare_epoch(
    method: AbstractEpochMethod,
    backend: AbstractTransitionBackend,
    distribution: jax.Array,
    *,
    frequency_map: AbstractFrequencyMap,
    generations: int,
    segregating: bool = False,
    immigration: jax.Array | None = None,
) -> EpochProblem:
    r"""Prepare a method's working problem from population-space inputs.

    **Arguments:**

    - `method`: Epoch algorithm whose representation requirement selects preparation.
    - `backend`: Prepared geometry, constructed outside JIT.
    - `distribution`: Full or interior mass, as in `AbstractEpochMethod.advance`.
    - `frequency_map`, `generations`, `segregating`, `immigration`: Process, static
      duration, state layout and per-generation source as in that interface.

    **Returns:**

    A PyTree implementing the read-only `EpochProblem` contract. Pass it to
    `method.solve`; custom `advance` implementations can use this function
    inside JIT or differentiation. Preparation preserves biological parameter
    links and owns source timing and representation-specific reconstruction.

    **Raises:**

    `TypeError` for invalid methods or unsupported combinations; `ValueError`
    for invalid static shapes/configuration. Runtime process-domain checks raise
    independently of the later solve's `throw` setting, even for zero generations.
    """
    if not isinstance(method, AbstractEpochMethod):
        raise TypeError("method must be an AbstractEpochMethod")
    method.validate(backend, frequency_map)
    if not method.allows_reduced_representation:
        return _prepare_transition(
            backend, distribution, frequency_map, generations, segregating, immigration
        )
    return _prepare_epoch(
        backend, distribution, frequency_map, generations, segregating, immigration
    )


@final
class Sequential(AbstractEpochMethod):
    r"""Repeat one-generation transitions without an additional power approximation.

    For the backend's represented transition $\widetilde P$ and initial mass
    column vector $v_0$, apply one transition per generation:

    $$v_{t+1}=\widetilde P^T v_t,\qquad
    v_g=\underbrace{\widetilde P^T\cdots\widetilde P^T}_{g\text{ applications}}v_0
       =(\widetilde P^T)^g v_0.$$

    Rows of $\widetilde P$ index source counts and columns index destination
    counts, so forward propagation uses its transpose. With immigration $b$,
    each step instead computes $v_{t+1}=\widetilde P^T v_t+b$; segregating
    evolution uses the interior restriction $\widetilde Q$.

    Construct as `Sequential()`. Geometry is supplied separately to `advance` or
    through `DemographicTransition`. The reported power-action error is zero;
    backend approximation and floating-point errors remain.
    """

    allows_reduced_representation: ClassVar[bool] = False

    def validate(
        self, backend: AbstractTransitionBackend, frequency_map: AbstractFrequencyMap
    ) -> None:
        r"""Check the types required by `AbstractEpochMethod.validate`.

        **Arguments:**

        - `backend`, `frequency_map`: Prepared transition geometry and process.

        **Returns:**

        `None`. No reduced-assembly map restrictions apply to Sequential.

        **Raises:**

        `TypeError` for objects outside the backend and frequency-map interfaces.
        """
        validate_representation(
            backend, frequency_map, reduced=False, method_name="Sequential"
        )

    def solve(self, problem: EpochProblem, *, throw: bool = True) -> EpochResult:
        r"""Iterate in working coordinates and reconstruct population mass.

        **Arguments:**

        - `problem`: Prepared problem; see `AbstractEpochMethod.solve`.
        - `throw`: Raise on nonfinite output when true.

        **Returns:**

        An `EpochResult` with zero power-approximation error. This does not
        measure accumulated backend error or floating-point roundoff.
        """
        state = _power_action(self, problem.operator, problem.initial, problem.steps)
        return problem.result(state, throw=throw)

    def _power_action(self, operator, initial, steps):
        """Apply repeated operator actions in working coordinates."""
        return jax.lax.fori_loop(0, steps, lambda _, state: operator.mv(state), initial)

    def _power_jvp(self, operator, tangent_operator, initial, tangent_initial, steps):
        """Evolve the state and its sensitivity without differentiating the loop."""
        if tangent_operator is None:
            return (
                _power_action(self, operator, initial, steps),
                _power_action(self, operator, tangent_initial, steps),
            )

        def step(_, carry):
            # Differentiate x_next = A x directly: dx_next = A dx + dA x.
            # Carrying the pair avoids differentiating a saved state trajectory.
            state, tangent = carry
            following = operator.mv(state)
            tangent = jax.tree.map(
                jnp.add, operator.mv(tangent), tangent_operator.mv(state)
            )
            return following, tangent

        return jax.lax.fori_loop(0, steps, step, (initial, tangent_initial))


Sequential.__init__.__doc__ = r"""Configure repeated one-generation evolution.

**Arguments:**

No arguments. Supply geometry separately to `advance` or `DemographicTransition`.

**Returns:**

`None`.
"""


class _AbstractKrylovMethod(AbstractEpochMethod):
    """Shared Krylov configuration, power orchestration, and sensitivities."""

    allows_reduced_representation: ClassVar[bool] = True
    _polynomial: eqx.AbstractClassVar[bool]

    dimension: int = 64
    rtol: float = 1e-8
    atol: float = 1e-12

    def __check_init__(self) -> None:
        """Validate static algorithm settings before preparing any geometry."""
        if (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, int)
            or self.dimension < 2
        ):
            raise ValueError("dimension must be a Python integer >= 2")
        # math validates Python configuration at construction; tolerances are
        # not traced model parameters or differentiable numerical inputs.
        if not all(
            math.isfinite(value) and value >= 0 for value in (self.rtol, self.atol)
        ):
            raise ValueError("rtol and atol must be finite and nonnegative")

    def validate(
        self, backend: AbstractTransitionBackend, frequency_map: AbstractFrequencyMap
    ) -> None:
        r"""Check backend and map compatibility for reduced power evolution.

        **Arguments:**

        - `backend`, `frequency_map`: Geometry and process as in
          `AbstractEpochMethod.validate`.

        **Returns:**

        `None`. Jacobi additionally checks that the map supports reduced assembly.

        **Raises:**

        `TypeError` for unsupported types or Jacobi map compositions. Parameter
        value checks are deferred to preparation, including for zero generations.
        """
        validate_representation(
            backend, frequency_map, reduced=True, method_name=type(self).__name__
        )

    def _power_action(self, operator, initial, steps):
        """Evaluate an unencoded power and require its numerical checks to pass."""
        return self.solve(_working_problem(operator, initial, steps)).distribution

    def _power_jvp(self, operator, tangent_operator, initial, tangent_initial, steps):
        """Approximate the mathematical power sensitivity without basis AD."""
        value = _power_action(self, operator, initial, steps)
        if tangent_operator is None:
            tangent = _power_action(self, operator, tangent_initial, steps)
        else:
            tangent = _frechet_action(
                self,
                operator,
                initial,
                tangent_operator.tangent,
                tangent_initial,
                steps,
            )
        return value, tangent

    def solve(self, problem: EpochProblem, *, throw: bool = True) -> EpochResult:
        r"""Approximate a prepared power and attach its mathematical sensitivities.

        **Arguments:**

        - `problem`: Prepared problem; see `AbstractEpochMethod.solve`.
        - `throw`: Raise on a failed primal action when true.

        **Returns:**

        An `EpochResult` with reconstructed mass and a power-error estimate from
        nested spaces and recurrence defects. This estimate is not a certificate
        and excludes backend approximation. With `throw=False`, inspect `converged`.

        **Raises:**

        Runtime errors for failed primal checks with `throw=True`, or failed
        sensitivities regardless of `throw`. Higher-order operator sensitivities
        raise explicitly; use Sequential when they are needed.
        """
        if not isinstance(problem, _PowerProblem):
            raise TypeError(
                "Krylov methods require a problem returned by prepare_epoch"
            )
        if problem.steps == 0:
            return _epoch_result(
                problem.reconstruct(problem.initial), jnp.asarray(0.0), self.atol, throw
            )
        arrays, static = eqx.partition(problem, eqx.is_array)
        # Basis construction is an adaptive approximation, not the derivative of
        # A**g x. Stop its AD path and attach that mathematical rule to the cached
        # value below, retaining the primal diagnostics without a second projection.
        raw_problem = eqx.combine(jax.tree.map(jax.lax.stop_gradient, arrays), static)
        projection = project(raw_problem, self.dimension, polynomial=self._polynomial)
        fine = _cached_power_action(
            self, problem.operator, problem.initial, problem.steps, projection.fine
        )
        return _projected_result(
            problem, projection._replace(fine=fine), self.rtol, self.atol, throw
        )


@final
class RationalKrylov(_AbstractKrylovMethod):
    r"""Approximate a prepared power action in a Krylov space built by shifted solves:

    $$
    R=[I+\tau(I-A)]^{-1},\qquad\tau=k/8,\qquad
    \mathcal K_m(R,z)=\operatorname{span}\{z,Rz,\ldots,R^{m-1}z\}.
    $$

    Here $A$ is the prepared forward operator, $z$ is the initial working state,
    and $k$ is the working exponent. An orthonormal basis $V_m$ for this space
    gives $A^kz\approx V_mH_m^kV_m^Tz$, where $H_m=V_m^TAV_m$.
    The shifted inverse builds the basis; the projected original operator
    supplies the power. Compared with polynomial Krylov, this space can capture
    slow, near-stationary dynamics with fewer basis vectors, reducing the cost
    of long intervals when the savings outweigh the shifted-solve cost.
    Jacobi uses a reusable dense LU factorization for shifted solves;
    other backends use unpreconditioned Lineax GMRES.
    Inner failures are reported as `linear_solve_failed`, independently of
    the outer power tolerances.

    !!! note "Theory"

        See [Krylov computation](../methods/jacobi_krylov.md#computation) for
        shifted solves and the projected power,
        [accuracy and controls](../methods/jacobi_krylov.md#accuracy-and-controls)
        for convergence, and [differentiation and adjoints](../methods/jacobi_krylov.md#differentiation-and-adjoints)
        for sensitivity calculations and the lack of a matched expectation transpose.
    """

    _polynomial: ClassVar[bool] = False


@final
class PolynomialKrylov(_AbstractKrylovMethod):
    r"""Approximate a prepared power action in a polynomial Krylov space:

    $$
    \mathcal K_m(A,z)=\operatorname{span}\{z,Az,\ldots,A^{m-1}z\},
    \qquad A^kz\approx V_mH_m^kV_m^Tz.
    $$

    Here $A$ is the prepared forward operator, $z$ is the initial working state,
    and $k$ is the working exponent. Arnoldi constructs an orthonormal basis
    $V_m$ for this space; $H_m=V_m^TAV_m$ is the projected operator. Building
    the basis needs only operator applications, with no shifted solves.
    All transition backends are supported, including Dense for small references.
    Basis storage is $O(dm)$ for working-state size $d$; the basis size is
    capped by `dimension`, $d$, and $k+1$.

    !!! note "Theory"

        See [Krylov computation](../methods/jacobi_krylov.md#computation) for
        the projected power, [accuracy and controls](../methods/jacobi_krylov.md#accuracy-and-controls)
        for convergence, and [differentiation and adjoints](../methods/jacobi_krylov.md#differentiation-and-adjoints)
        for sensitivity calculations and the lack of a matched expectation transpose.
    """

    _polynomial: ClassVar[bool] = True


_krylov_init_doc = r"""Configure Krylov powering independently of geometry.

**Arguments:**

- `dimension`: Maximum Krylov basis size, a Python integer at least two;
  capped by the prepared working-state dimension.
- `rtol`, `atol`: Finite nonnegative relative and absolute tolerances for the
  power-action error estimate; these do not control backend approximation.

**Returns:**

`None`. Supply transition geometry separately.

**Raises:**

`ValueError` for invalid dimension or tolerances.
"""
RationalKrylov.__init__.__doc__ = _krylov_init_doc
PolynomialKrylov.__init__.__doc__ = _krylov_init_doc
