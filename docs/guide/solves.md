# Expected visits

For a square transition, `expected_visits()` computes the expected number of
visits to each segregating state, starting from one mutant copy. It includes
generation zero and stops at the first endpoint hit.

```python
import jax
import drifft

jax.config.update("jax_enable_x64", True)
transition = drifft.Transition(drifft.FFT(64), drifft.Underdominance(0.01))
result = transition.expected_visits()
if not bool(result.converged):
    raise RuntimeError(f"{result.status}: residual={result.residual_norm}")
visits = result.visits
```

## Solver settings and limitations

Choose solver tolerances for the residual you need, then check `converged`
before interpreting the visit vector. Long absorption times can amplify errors
in the underlying transition even when the solve converges.

Choose `solver="gmres"` (default) or `"bicgstab"`; adjust `rtol`, `atol`,
and iteration limits as needed. The [transition API](../api/transitions.md)
lists defaults and available settings.

`result.converged` requires a successful Lineax status, finite output, and a
fresh infinity-norm residual no larger than `atol+rtol`. Always check it.
`result.stats` preserves the installed Lineax solver's iteration counters; GMRES
`num_steps` is not a count of individual matvecs. Failures retain the candidate
iterate for diagnosis, but it must not be treated as a valid occupation vector.

Compare visits and total absorption time under tighter backend settings,
using the [accuracy guide](numerics.md). The
[model chapter](model.md#interior-states-and-expected-visits) derives the
occupation equation and its transience requirement.
