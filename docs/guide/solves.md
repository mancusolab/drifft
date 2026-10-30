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
visits = result.visits
```

## Solver settings and limitations

Numerical failure raises by default. Use `throw=False` to inspect a failed
candidate and check `converged` before interpreting its visit vector:

```python
result = transition.expected_visits(throw=False)
if not bool(result.converged):
    print(result.status, result.solver_status, result.residual_norm)
```

Under `vmap`, inspect each result with `throw=False`; throwing may fail the
whole batch. Long absorption times can amplify errors
in the underlying transition even when the solve converges.

Choose `solver="gmres"` (default) or `"bicgstab"`; adjust `rtol`, `atol`,
and iteration limits as needed. The [transition API](../api/transitions.md)
lists defaults and available settings.

`result.status` is an `ExpectedVisitsStatus`: successful, linear-solve failure,
nonfinite output, or an unacceptable residual. Success requires a successful
Lineax solve, finite output and residual, and a fresh infinity-norm residual no
larger than `atol+rtol`. `converged` is true exactly when this status is successful.
`result.solver_status` preserves the underlying Lineax result code.
`result.stats` preserves the installed Lineax solver's iteration counters; GMRES
`num_steps` is not a count of individual matvecs. Failures retain the candidate
iterate for diagnosis, but it must not be treated as a valid occupation vector.

Compare visits and total absorption time under tighter backend settings,
using the [accuracy guide](numerics.md). The
[model chapter](model.md#interior-states-and-expected-visits) derives the
occupation equation and its transience requirement.
