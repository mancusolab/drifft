# Direct epoch operations

The [population-history guide](demography.md) covers the usual workflow.
For experiments or custom orchestration, an evolution method can also act
directly on a prepared backend, including arbitrary immigration vectors. This
page shows the lower-level call for a single constant-size interval; the
[Krylov chapter](../methods/jacobi_krylov.md) explains the power algorithms.

For example:

```python
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)
backend = drifft.Jacobi(128)
initial = jnp.zeros(127).at[0].set(1.0)
method = drifft.RationalKrylov(dimension=40)
source = jnp.zeros_like(initial).at[0].set(128 * 1e-8)
result = method.advance(
    backend,
    initial,
    frequency_map=drifft.Neutral(),
    generations=250,
    segregating=True,
    immigration=source,
)
final = result.distribution
```

This applies the hold backend for all 250 generations.
A `Demography([64, 64], [250])` interval instead uses 249 hold generations and one
boundary generation using the configured boundary backend, following
the history's interval convention even when the sizes are equal. They use
different approximations to the same biological recurrence and can produce
different results.

## Mass and immigration

The state layout determines whether endpoint mass is retained. Immigration
is a separate source added after the transition, so its timing matters.

Full-state input has `gene_copies + 1` entries, including count zero and fixation.
With `segregating=True`, input has `gene_copies - 1` entries and each generation
discards boundary mass. `immigration` adds the supplied vector **after every
generation**, including the last. Zero generations returns the original mass
without immigration. Signed vectors are allowed; outputs are never clipped or
renormalized.

For a singleton mutation source, the demographic API supplies the scaling
automatically; see [segregating mass and mutation input](demography.md#segregating-mass-and-mutation-input).
For recurrent mutation of existing alleles, use a [frequency map](frequency_maps.md#add-mutation).

## Accuracy and failures

A direct method call returns diagnostics for its power approximation.
Check them before using a result, and validate backend accuracy separately.

By default, numerical failure raises. Set `throw=False` to inspect
`result.converged`, `result.status`, and `result.error_estimate`; domain errors
still raise. A failing member under `vmap` may fail the batch.

These diagnostics assess the power approximation only. Follow the
[accuracy guide](numerics.md) to check backend error and gradients separately;
the [Krylov chapter](../methods/jacobi_krylov.md#accuracy-and-controls) defines
the acceptance criterion.

For a multi-interval calculation, use the [population-history interface](demography.md).
It handles size-changing boundary steps and singleton-source scaling. The
[autodiff contract](../methods/autodiff.md#accuracy-and-limitations) applies to
both direct and demographic calls.

## Custom evolution methods

Implement `AbstractEpochMethod` to supply a different power algorithm. Public
preparation handles representation and source timing; the method computes the
prepared power and returns checked population mass. This example uses ordinary
JAX differentiation through a fixed loop; custom methods own their derivative
and batching behavior.

```python
from typing import ClassVar, final
import equinox as eqx

@final
class RepeatedTransition(drifft.AbstractEpochMethod):
    """Evolve in population coordinates using repeated transition actions."""

    uses_reduced_representation: ClassVar[bool] = False

    def validate(self, backend, frequency_map):
        """Accept the same representation/process combinations as Sequential."""
        # Reuse Sequential's supported backend/map contract through composition.
        drifft.Sequential().validate(backend, frequency_map)

    def solve(self, problem: drifft.EpochProblem, *, throw=True):
        """Apply the prepared power and reconstruct checked population mass."""
        state = jax.lax.fori_loop(
            0, problem.steps,
            lambda _, x: problem.operator.mv(x),
            problem.initial,
        )
        return problem.result(state, throw=throw)

    @eqx.filter_jit
    def advance(self, backend, distribution, *, throw=True, **kwargs):
        """Prepare population inputs using the supported public interface."""
        problem = drifft.prepare_epoch(self, backend, distribution, **kwargs)
        return self.solve(problem, throw=throw)

method = RepeatedTransition()
result = method.advance(
    backend, initial, frequency_map=drifft.Neutral(), generations=10,
    segregating=True,
)
```

The same method can be passed to `DemographicTransition`. Declare
`uses_reduced_representation=True` only if the algorithm can consume the
backend's reduced working coordinates. Always use the prepared `steps` and
`problem.result`: reduction can absorb a generation into encoding, and source
terms can add an auxiliary coordinate or a final offset. `problem.error_norm`
maps a working-state error into the acceptance norm for population mass.
Supply an absolute `error_estimate` and `tolerance` to `result` when the method
approximates the power. These checks exclude backend approximation error.

`EpochProblem` describes the opaque prepared PyTree returned by `prepare_epoch`;
it does not expose projection bookkeeping or support direct construction.
Matched demographic expectations remain available only for Sequential.
