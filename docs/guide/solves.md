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

The solver choices are `solver="gmres"` (default) and `"bicgstab"`. Defaults are
`rtol=1e-9`, `atol=1e-11`, `max_steps=2000`, `restart=50` (GMRES only). There is
no conjugate-gradient option: the transition need not be symmetric. The solver
uses the interior transition without renormalizing it. Finite visits require
eventual endpoint hitting.

`result.converged` requires a successful Lineax status, finite output, and a
fresh infinity-norm residual no larger than `atol+rtol`. Always check it.
`result.stats` preserves the installed Lineax solver's iteration counters; GMRES
`num_steps` is not a count of individual matvecs. Failures retain the candidate
iterate for diagnosis, but it must not be treated as a valid occupation vector.

A small residual establishes convergence for the approximate transition.
Long absorption times can amplify approximation errors, especially near
neutrality. Compare visits and total absorption time under tighter backend
settings; use a dense reference for small populations. Check gradients separately
when using visits in an inference objective. Convergence alone does not certify
accuracy for the exact Wright–Fisher model.
