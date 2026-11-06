# Custom evolution methods

An evolution method controls how `drifft` propagates a distribution over a
constant-size interval. Implement one when you want a different algorithm for
applying many generations of the same transition. Changing selection or mutation
belongs in a [frequency map](api/frequency_maps.md); changing the one-generation
numerical representation belongs in a [backend](api/backends.md).

A custom method subclasses [`AbstractEpochMethod`](api/methods.md#drifft.AbstractEpochMethod) and implements two methods:

- `validate(backend, frequency_map)` rejects unsupported types or combinations.
  It runs before preparation and should not inspect traced parameter values.
- `solve(problem, *, throw=True)` computes the prepared power and returns an
  [`EpochResult`](api/methods.md#drifft.EpochResult) using `problem.result(...)`.

The inherited `advance` connects these methods: it calls [`prepare_epoch`](api/methods.md#drifft.prepare_epoch), then
`solve`. Preparation handles state restriction, source mass, and backend-specific
coordinates, so the power algorithm can work with a single linear operator.

## Implement dense matrix powering

This example materializes the prepared operator and computes its power using
dense matrix multiplication. It exists solely to demonstrate how to extend the
API with a different numerical algorithm. Run the blocks on this page in order.

```python
from typing import ClassVar, final

import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)


@final
class DensePower(drifft.AbstractEpochMethod):
    """Compute the prepared power by materializing a dense matrix.

    This teaching example illustrates a new algorithm behind the epoch-method
    interface. Preparation and reconstruction retain ownership of state layout.
    """

    allows_reduced_representation: ClassVar[bool] = False

    def validate(self, backend, frequency_map):
        """Require transition geometry and a frequency map for population coordinates."""
        # Check interfaces, not parameter values that may be traced by JAX.
        if not isinstance(backend, drifft.AbstractTransitionBackend):
            raise TypeError("DensePower requires an AbstractTransitionBackend")
        if not isinstance(frequency_map, drifft.AbstractFrequencyMap):
            raise TypeError("frequency_map must be an AbstractFrequencyMap")

    def solve(self, problem: drifft.EpochProblem, *, throw=True):
        """Apply the prepared power and reconstruct checked population mass."""
        # For illustration only: dense powering uses O(n**2) storage and
        # O(n**3) work per matrix product, making realistic problems costly.
        # Preparation supplies the exponent, including generation accounting.
        matrix = problem.operator.as_matrix()
        state = jnp.linalg.matrix_power(matrix, problem.steps) @ problem.initial
        # Decode the working state and apply the shared numerical checks.
        return problem.result(state, throw=throw)


pop_size = 64
backend = drifft.FFT(pop_size=pop_size)
# Interior count one represents a single copy of the allele.
initial = jnp.zeros(2 * pop_size - 1).at[0].set(1.0)
method = DensePower()
result = method.advance(
    backend,
    initial,
    frequency_map=drifft.Neutral(),
    generations=10,
    segregating=True,
)
final_mass = result.distribution
```

The result contains mass at counts 1 through `2 * pop_size - 1`. Because this
call discards loss and fixation at every generation, its sum is the probability
that the allele remains segregating after ten generations. No renormalization
is applied.

## Work with the prepared problem

For a prepared operator $A$, initial working vector $z_0$, and exponent $k$,
`solve` computes $z_k=A^k z_0$. These are `problem.operator`, `problem.initial`,
and `problem.steps`. Algorithms can use `operator.mv(state)` without storing
a dense matrix. This example deliberately uses `operator.as_matrix()` to
illustrate a different computation strategy.

Declare `allows_reduced_representation=False` when your algorithm requires
population coordinates. `True` permits preparation to use a backend's reduced
coordinates, such as Jacobi coefficients. It does not guarantee a smaller
problem for every backend. The validation method must also accept the
backend/map combinations that the chosen representation supports. This example
requests population coordinates and checks the public backend and frequency-map
interfaces directly; it needs no additional restrictions for reduced assembly.

Even with population coordinates, source mass can append an auxiliary
coordinate to turn an affine recurrence into a linear action. With reduced
coordinates, encoding can incorporate the first generation, leaving
`problem.steps` smaller than the requested generation count. Reconstruction
can also add a final source contribution. Consequently:

- Use `problem.initial.shape` for working-vector allocations.
- Use `problem.steps` for the exponent, including when it is zero.
- Finish with `problem.result(state, throw=throw)` to recover population mass
  and apply the shared numerical checks.

Calling `problem.reconstruct(state)` alone converts coordinates but does not
check convergence. Do not normalize the reconstructed mass: segregating mass
can decrease, source mass can increase it, and signed inputs are supported.

## Report approximation error

Dense matrix powering introduces no additional power approximation, so the
example uses the default zero error estimate. This does not claim that the
backend is exact or that floating-point roundoff is absent.

An approximate power algorithm must supply an absolute population-space L1
error estimate and an absolute acceptance tolerance to `problem.result`.
If the algorithm estimates a working-state error vector `delta`, use
`problem.error_norm(delta)` to measure it in the required norm, including any
auxiliary-coordinate defect. A small residual is not automatically an error
bound; deriving the estimate remains the method's responsibility.

With `throw=True`, a nonfinite output or rejected error estimate raises.
With `throw=False`, the result retains the candidate distribution and reports
failure through `status` and `converged`. Invalid shapes or invalid tolerances
still raise. These checks cover the power algorithm, not backend or
representation accuracy.

## Use the method in a demographic history

Pass the method through the same argument used for built-in algorithms:

```python
history = drifft.Demography(pop_sizes=[64, 32], generations=[10])
geometry = history.prepare(backend_factory=drifft.FFT)
model = drifft.DemographicTransition(
    geometry,
    frequency_map=drifft.Neutral(),
    method=method,
)
history_result = model.advance(initial, segregating=True)
```

Here the custom method handles nine generations at size 64. The final generation
changes the size to 32 and is handled by the boundary transition. Demographic
holds call the method's `advance`, including an override if you supply one.
Keep the inherited implementation unless the method needs custom orchestration.

[`DemographicTransition.expect`](api/demography.md#drifft.DemographicTransition.expect) calls `solve` on the transposed working operator, with the encoding and
decoding maps transposed around it. A custom solver must support this action
and retain the represented operator semantics.

## Check a new method

Compare forward results with [`Sequential`](api/methods.md#drifft.Sequential) using the same backend and frequency
map. Include zero and one generation, full and segregating distributions, and
source mass: these expose mistakes in exponents, state layout, and source timing.
For methods that allow reduction, also exercise a reduced Jacobi problem.

Custom methods own their differentiation and batching behavior. The example
uses JAX differentiation through matrix construction and powering; subclassing does not install
the built-in methods' custom derivative rules. Check the transformations your
method supports, including gradients through frequency-map parameters and
batched distributions. Exercise numerical failure with both `throw` settings
if the method approximates the power.

See the [method API](api/methods.md#drifft.AbstractEpochMethod) for the shared
`validate`, `solve`, and `advance` contracts, and the
[prepared-problem reference](api/methods.md#implementing-custom-methods) for
[`prepare_epoch`](api/methods.md#drifft.prepare_epoch) and [`EpochProblem`](api/methods.md#drifft.EpochProblem).
