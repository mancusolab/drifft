r"""Evaluate prepared epoch powers and reconstruct population-space results.

An epoch method receives a working operator $A$, seed $z_0$, exponent $k$ and
reconstruction maps from `preparation.py`/`problem.py`. Its task is to evaluate
$A^k z_0$ and return a checked population result. Separating this task from
geometry preparation lets the same numerical algorithm operate on different
transition representations.

[`Sequential`][drifft.Sequential] applies the represented transition repeatedly.
[`PolynomialKrylov`][drifft.PolynomialKrylov] and
[`RationalKrylov`][drifft.RationalKrylov] approximate the power in a projected
space. For built-in Krylov methods, numerical construction is stopped for AD,
and a cached power binding attaches the mathematical derivative to the computed
value. This preserves primal diagnostics without differentiating basis selection
or repeating the primal projection.

Numerical representation selection is delegated to `operators.py`; cached AD
binding is delegated to `autodiff.py`, which distinguishes ordinary operator
PyTrees from Jacobi parameter-space values. Private method capabilities here
select coupled-recurrence or projected sensitivity evaluation and enforce its
numerical acceptance policy. The primitives retain the mathematical derivative
and transpose contracts. Custom methods define their own differentiation
behavior through the prepared-problem interface; these private capabilities
are not additional requirements on `AbstractEpochMethod`.

Thus numerical power evaluation lives here and in `projection.py`, mathematical
power transformation rules in `autodiff.py`, and sensitivity calculations
in `sensitivity.py`. Parameter-dependent preparation and reconstruction remain
outside the power primitive and retain their derivatives in the full epoch.
"""

# pattern: Functional Core
import math
from abc import abstractmethod
from typing import ClassVar, final

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from .._backend import AbstractTransitionBackend, Jacobi
from .._frequency import AbstractFrequencyMap
from .autodiff import _cached_power_action, _frechet_action, _power_action
from .operators import _numerical_operator, _operator_pushforward
from .preparation import (
    _prepare_epoch,
    validate_domain,
    validate_representation,
)
from .problem import (
    EpochProblem,
    EpochResult,
    PowerResult,
    _epoch_source,
    _identity,
    _working_problem,
)
from .projection import _projected_result, project
from .sensitivity import (
    _frechet_adjoint_factors,
    _frechet_power,
    _sequential_power_pair,
)


class AbstractEpochMethod(eqx.Module):
    r"""An algorithm for a prepared linear power action, independent of geometry.

    Implement `validate` and `solve`. The default `advance` prepares a reusable
    representation, encodes mass, solves the working power, and reconstructs mass.
    """

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

        `TypeError` for unsupported backend or map types. [`Sequential`][drifft.Sequential] accepts
        all transition backends; reduced [`Jacobi`][drifft.Jacobi] powers restrict map compositions.
        """
        ...

    @abstractmethod
    def solve(self, problem: EpochProblem, initial: jax.Array) -> PowerResult:
        r"""Evaluate the prepared operator power on an encoded initial vector.

        **Arguments:**

        - `problem`: An [`EpochProblem`][drifft.EpochProblem] from [`prepare_epoch`][drifft.prepare_epoch].
          Its `steps` field gives the exponent in working coordinates.
        - `initial`: Float64 working vector returned by `problem.encode(mass)`.

        **Returns:**

        A [`PowerResult`][drifft.PowerResult] containing the final working vector
        and solver diagnostics. Call `problem.result(solution, throw=...)` to
        reconstruct population mass and check convergence.
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
        source_mass: jax.Array | None = None,
        throw: bool = True,
    ) -> EpochResult:
        r"""Bind geometry and evolve a constant-size interval.

        !!! example "Example: Evolve with singleton mutation input"

            ```python
            import jax
            import jax.numpy as jnp
            import drifft

            jax.config.update("jax_enable_x64", True)
            backend = drifft.Jacobi(pop_size=64)
            initial = jnp.zeros(127).at[0].set(1.0)
            source = jnp.zeros_like(initial).at[0].set(128 * 1e-8)
            result = drifft.RationalKrylov(dimension=40).advance(
                backend, initial, frequency_map=drifft.Neutral(),
                generations=250, segregating=True, source_mass=source,
            )
            final = result.distribution
            ```

            The source adds singleton mass after every generation, including
            the last. For automatic per-copy mutation-rate scaling across
            changing sizes, see [new mutations through time](../examples/mutation_input.md).

        **Arguments:**

        - `backend`: Prepared geometry compatible with this method.
        - `distribution`: Real mass, converted to float64. Length is
          `2 * backend.pop_size + 1`, or `2 * backend.pop_size - 1` if segregating.
          Signed entries are supported; no normalization is applied.
        - `frequency_map`: Process shared by every generation.
        - `generations`: Nonnegative Python integer, static under JIT.
        - `segregating`: Discard endpoint mass at each generation.
        - `source_mass`: Source mass added after every generation, including the
          last, matching the input shape; `None` means no source. This can
          represent new mutations by placing mass at count one, as in the example.
          In segregating mode it is added after endpoint mass is discarded.
        - `throw`: Raise on numerical failure; under vmap this may fail the whole batch.

        **Returns:**

        An [`EpochResult`][drifft.EpochResult] in population coordinates; zero generations preserves
        the input without adding source mass. [`Sequential`][drifft.Sequential] reports zero power-approximation
        error; Krylov estimates exclude backend and representation errors.
        Check `converged` with `throw=False`.

        **Raises:**

        `TypeError` for unsupported combinations; `ValueError` for static input errors.
        Runtime domain checks remain active even with `throw=False`; Krylov
        sensitivity numerical failures also raise independently of `throw`; their
        extra error-estimate checks are opt-in via `check_sensitivity=True`.
        [`Sequential`][drifft.Sequential]
        supports higher derivatives; higher-order Krylov operator sensitivities
        are unsupported.

        !!! note "Generation timing"

            This call uses the supplied backend for all 250 generations.
            [`Demography([64, 64], [250])`][drifft.Demography] instead uses 249 hold generations and
            one boundary generation, even with equal sizes. Different numerical
            representations of these steps can produce different approximations
            to the same biological recurrence.
        """
        problem = prepare_epoch(
            self,
            backend,
            frequency_map=frequency_map,
            generations=generations,
            segregating=segregating,
            source_mass=source_mass,
        )
        initial = problem.encode(distribution)
        solution = (
            PowerResult(initial) if generations == 0 else self.solve(problem, initial)
        )
        return problem.result(solution, throw=throw)


def prepare_epoch(
    method: AbstractEpochMethod,
    backend: AbstractTransitionBackend,
    *,
    frequency_map: AbstractFrequencyMap,
    generations: int,
    segregating: bool = False,
    source_mass: jax.Array | None = None,
) -> EpochProblem:
    r"""Prepare a method's working problem from population-space inputs.

    Selects working coordinates from the backend and frequency map after method
    validation. The problem can encode multiple initial distributions. The returned operator
    includes per-generation source mass when supplied; reconstruction restores
    the requested population layout and any final source contribution.

    !!! example "Example: Prepare and solve an interval"

        ```python
        # Split the preparation and solve stages of method.advance.
        problem = drifft.prepare_epoch(method, backend,
            frequency_map=drifft.Neutral(), generations=10, segregating=True)
        result = problem.result(method.solve(problem, problem.encode(initial)))
        ```

    **Arguments:**

    - `method`: Epoch algorithm used for compatibility validation.
    - `backend`: Prepared geometry, constructed outside JIT.
    - `frequency_map`, `generations`, `segregating`, `source_mass`: Process, static
      duration, state layout and per-generation source as in that interface.

    **Returns:**

    A PyTree implementing the read-only [`EpochProblem`][drifft.EpochProblem] contract. Pass it to
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
    source = _epoch_source(source_mass, backend._gene_copies, generations, segregating)
    if generations == 0:
        offset = validate_domain(backend, jnp.zeros_like(source), frequency_map)
        operator = lx.IdentityLinearOperator(
            jax.ShapeDtypeStruct(offset.shape, offset.dtype)
        )
        return EpochProblem(
            operator, _identity, 0, _identity, offset, False, jnp.asarray(1.0)
        )
    return _prepare_epoch(
        backend,
        frequency_map,
        generations,
        segregating,
        source,
        source_mass is not None,
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
    counts, so forward propagation uses its transpose. With source mass $b$,
    each step instead computes $v_{t+1}=\widetilde P^T v_t+b$; segregating
    evolution uses the interior restriction $\widetilde Q$.

    Construct as [`Sequential()`][drifft.Sequential]. Geometry is supplied separately to `advance` or
    through [`DemographicTransition`][drifft.DemographicTransition]. The reported power-action error is zero;
    backend approximation and floating-point errors remain.
    """

    def validate(
        self, backend: AbstractTransitionBackend, frequency_map: AbstractFrequencyMap
    ) -> None:
        r"""Check the types required by [`AbstractEpochMethod.validate`][drifft.AbstractEpochMethod.validate].

        **Arguments:**

        - `backend`, `frequency_map`: Prepared transition geometry and process.

        **Returns:**

        `None`. No reduced-assembly map restrictions apply to [`Sequential`][drifft.Sequential].

        **Raises:**

        `TypeError` for objects outside the backend and frequency-map interfaces.
        """
        validate_representation(backend, frequency_map, method_name="Sequential")

    def solve(self, problem: EpochProblem, initial: jax.Array) -> PowerResult:
        r"""Apply the prepared operator successively in working coordinates.

        **Arguments:**

        - `problem`: Prepared epoch supplying the operator and step count.
        - `initial`: Encoded initial vector for this epoch.

        **Returns:**

        A [`PowerResult`][drifft.PowerResult] with the final vector and zero
        power-approximation error. A zero exponent returns the initial vector.
        """
        operator = _numerical_operator(problem.operator)
        return PowerResult(_power_action(self, operator, initial, problem.steps))

    def _power_action(self, operator, initial, steps):
        """Apply repeated operator actions in working coordinates."""
        return jax.lax.fori_loop(0, steps, lambda _, state: operator.mv(state), initial)

    def _linearize_power(self, operator, initial, operator_direction, tangent, steps):
        """Fuse primal and tangent recurrence while retaining ordinary higher AD."""
        return _sequential_power_pair(
            operator,
            _operator_pushforward(operator, operator_direction),
            initial,
            tangent,
            steps,
        )


Sequential.__init__.__doc__ = r"""Configure repeated one-generation evolution.

**Arguments:**

No arguments. Supply geometry separately to `advance` or [`DemographicTransition`][drifft.DemographicTransition].
"""


class AbstractKrylovMethod(AbstractEpochMethod):
    r"""Shared interface for polynomial and rational [Krylov](https://en.wikipedia.org/wiki/Krylov_subspace) power approximations.

    Krylov methods approximate a prepared power $A^k z$ in a smaller subspace,
    then reconstruct population mass. The concrete method determines how the
    subspace is built; this class supplies configuration, compatibility checks,
    convergence checks, and differentiation of the mathematical power action.

    Backend reduction is permitted when available. Power-error estimates exclude
    backend and representation errors. First-order operator sensitivities are
    supported; higher-order operator sensitivities require [`Sequential`][drifft.Sequential].

    Use [`PolynomialKrylov`][drifft.PolynomialKrylov] or [`RationalKrylov`][drifft.RationalKrylov] for evolution. For an independent
    power algorithm, implement [`AbstractEpochMethod`][drifft.AbstractEpochMethod]; this base shares the
    built-in Krylov machinery rather than providing configurable basis builders
    or shifted-solve strategies.
    """

    _polynomial: eqx.AbstractClassVar[bool]

    dimension: int = 64
    rtol: float = 1e-8
    atol: float = 1e-12
    check_sensitivity: bool = eqx.field(static=True, default=False)

    def __check_init__(self) -> None:
        """Validate static algorithm settings before preparing any geometry."""
        if not isinstance(self.check_sensitivity, bool):
            raise ValueError("check_sensitivity must be a Python bool")
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
          [`AbstractEpochMethod.validate`][drifft.AbstractEpochMethod.validate].

        **Returns:**

        `None`. [`Jacobi`][drifft.Jacobi] additionally checks that the map supports reduced assembly.

        **Raises:**

        `TypeError` for unsupported types or [`Jacobi`][drifft.Jacobi] map compositions. Parameter
        value checks are deferred to preparation, including for zero generations.
        """
        validate_representation(backend, frequency_map, method_name=type(self).__name__)
        if isinstance(backend, Jacobi):
            backend._validate_reduced_map(frequency_map)

    def _power_action(self, operator, initial, steps):
        """Evaluate an uncached power on an initial-state tangent or cotangent."""
        problem = _working_problem(operator, steps)
        result = problem.result(self.solve(problem, initial), throw=False)
        return eqx.error_if(
            result.distribution,
            ~result.converged,
            "Krylov sensitivity action did not converge for the state direction",
        )

    def _linearize_power(self, operator, initial, operator_direction, tangent, steps):
        """Retain the cached primal and bind a mathematically transposable direction."""
        return None, _frechet_action(
            self, operator, initial, operator_direction, tangent, steps
        )

    def _sensitivity_action(self, operator, perturbation, initial, tangent, steps):
        """Check numerical failures and optionally the sensitivity error estimate."""
        result = _frechet_power(
            operator,
            perturbation,
            initial,
            tangent,
            steps,
            dimension=self.dimension,
            rtol=self.rtol,
            atol=self.atol,
            polynomial=self._polynomial,
            check_error=self.check_sensitivity,
        )
        return eqx.error_if(
            result.distribution,
            ~result.converged,
            "Krylov sensitivity action did not converge; increase dimension or use Sequential",
        )

    def _sensitivity_adjoint_factors(self, operator, initial, cotangent, steps):
        """Return checked operator-adjoint factors for representation-side contraction."""
        left, right, _, converged = _frechet_adjoint_factors(
            operator,
            initial,
            cotangent,
            steps,
            dimension=self.dimension,
            rtol=self.rtol,
            atol=self.atol,
            polynomial=self._polynomial,
            check_error=self.check_sensitivity,
        )
        left = eqx.error_if(
            left,
            ~converged,
            "Krylov operator pullback did not converge; increase dimension or use Sequential",
        )
        return left, right

    def solve(self, problem: EpochProblem, initial: jax.Array) -> PowerResult:
        r"""Approximate the prepared power in a Krylov space.

        **Arguments:**

        - `problem`: Prepared epoch supplying the operator, exponent, and error norm.
        - `initial`: Encoded initial vector used to build the Krylov basis.

        **Returns:**

        A [`PowerResult`][drifft.PowerResult] with the working vector, a population-space
        error estimate, acceptance tolerances, and inner-solve success flag.

        **Raises:**

        `TypeError` if `problem` is not an [`EpochProblem`][drifft.EpochProblem].
        """
        if not isinstance(problem, EpochProblem):
            raise TypeError(
                "Krylov methods require a problem returned by prepare_epoch"
            )
        if problem.steps == 0:
            return PowerResult(initial)
        arrays, static = eqx.partition(problem, eqx.is_array)
        # Basis construction is an adaptive approximation, not the derivative of
        # A**g x. Stop its AD path and attach that mathematical rule to the cached
        # value below, retaining the primal diagnostics without a second projection.
        raw_problem = eqx.combine(jax.tree.map(jax.lax.stop_gradient, arrays), static)
        raw_problem = eqx.tree_at(
            lambda p: p.operator, raw_problem, _numerical_operator(raw_problem.operator)
        )
        projection = project(
            raw_problem,
            jax.lax.stop_gradient(initial),
            self.dimension,
            polynomial=self._polynomial,
        )
        fine = _cached_power_action(
            problem.operator,
            self,
            initial,
            problem.steps,
            projection.fine,
            raw_problem.operator,
        )
        return _projected_result(
            problem, projection._replace(fine=fine), self.rtol, self.atol
        )


@final
class RationalKrylov(AbstractKrylovMethod):
    r"""Approximate a prepared power action in a [Krylov space](https://en.wikipedia.org/wiki/Krylov_subspace) built by shifted solves:

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
    [`Jacobi`][drifft.Jacobi] uses a reusable dense LU factorization for shifted solves;
    other backends use unpreconditioned Lineax GMRES.
    Inner failures are reported as `linear_solve_failed`, independently of
    the outer power tolerances.

    !!! note "Theory"

        See [Krylov computation](../methods/jacobi_krylov.md#computation) for
        shifted solves and the projected power,
        [accuracy and controls](../methods/jacobi_krylov.md#accuracy-and-controls)
        for convergence, and [differentiation and adjoints](../methods/jacobi_krylov.md#differentiation-and-adjoints)
        for sensitivity calculations and independently approximated backward actions.
    """

    _polynomial: ClassVar[bool] = False


@final
class PolynomialKrylov(AbstractKrylovMethod):
    r"""Approximate a prepared power action in a polynomial [Krylov space](https://en.wikipedia.org/wiki/Krylov_subspace):

    $$
    \mathcal K_m(A,z)=\operatorname{span}\{z,Az,\ldots,A^{m-1}z\},
    \qquad A^kz\approx V_mH_m^kV_m^Tz.
    $$

    Here $A$ is the prepared forward operator, $z$ is the initial working state,
    and $k$ is the working exponent. Arnoldi constructs an orthonormal basis
    $V_m$ for this space; $H_m=V_m^TAV_m$ is the projected operator. Building
    the basis needs only operator applications, with no shifted solves.
    All transition backends are supported, including [`Dense`][drifft.Dense] for small references.
    Basis storage is $O(dm)$ for working-state size $d$; the basis size is
    capped by `dimension`, $d$, and $k+1$.

    !!! note "Theory"

        See [Krylov computation](../methods/jacobi_krylov.md#computation) for
        the projected power, [accuracy and controls](../methods/jacobi_krylov.md#accuracy-and-controls)
        for convergence, and [differentiation and adjoints](../methods/jacobi_krylov.md#differentiation-and-adjoints)
        for sensitivity calculations and independently approximated backward actions.
    """

    _polynomial: ClassVar[bool] = True


_krylov_init_doc = r"""Configure Krylov powering independently of geometry.

Supply transition geometry separately.

**Arguments:**

- `dimension`: Maximum Krylov basis size, a Python integer at least two;
  capped by the prepared working-state dimension.
- `rtol`, `atol`: Finite nonnegative relative and absolute tolerances for the
  power-action error estimate; these do not control backend approximation.
- `check_sensitivity`: Static Python bool, default `False`. When true, also
  require operator-sensitivity and operator-pullback error estimates to satisfy
  these tolerances. Inner-solve failures and nonfinite derivative outputs always
  raise, independently of `throw`. Disabling the heuristic does not certify
  derivative accuracy or disable ordinary power-action convergence checks.

**Raises:**

`ValueError` for invalid dimension, tolerances, or a nonboolean sensitivity flag.
"""
AbstractKrylovMethod.__init__.__doc__ = _krylov_init_doc
RationalKrylov.__init__.__doc__ = _krylov_init_doc
PolynomialKrylov.__init__.__doc__ = _krylov_init_doc
