# Custom evolution methods

Implement `validate` and `solve` on `AbstractEpochMethod` to supply a different
power algorithm. The inherited `advance` handles preparation, including
representation and source timing; `solve` computes the prepared power and
returns checked population mass. Override `advance` only for custom orchestration.
This example uses ordinary
JAX differentiation through a fixed loop; custom methods own their derivative
and batching behavior.

```python
from typing import ClassVar, final

import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)
backend = drifft.FFT(128)
initial = jnp.zeros(127).at[0].set(1.0)

@final
class RepeatedTransition(drifft.AbstractEpochMethod):
    """Evolve in population coordinates using repeated transition actions."""

    allows_reduced_representation: ClassVar[bool] = False

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

method = RepeatedTransition()
result = method.advance(
    backend, initial, frequency_map=drifft.Neutral(), generations=10,
    segregating=True,
)
```

The same method can be passed to `DemographicTransition`. Declare
`allows_reduced_representation=True` only if the algorithm can consume the
backend's reduced working coordinates. Always use the prepared `steps` and
`problem.result`: reduction can absorb a generation into encoding, and source
terms can add an auxiliary coordinate or a final offset. `problem.error_norm`
maps a working-state error into the acceptance norm for population mass.
Supply an absolute `error_estimate` and `tolerance` to `result` when the method
approximates the power. These checks exclude backend approximation error.

`EpochProblem` describes the opaque prepared PyTree returned by `prepare_epoch`;
it does not expose projection bookkeeping or support direct construction.
Matched demographic expectations remain available only for Sequential.

See the [extension API](api/extensions.md) for prepared power problems and
[evolution methods](api/methods.md#drifft.AbstractEpochMethod) for the method contract.
