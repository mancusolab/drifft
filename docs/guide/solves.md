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
`rtol=1e-9`, `atol=1e-11`, `max_steps=2000`, `restart=50` (GMRES only). There are
no symmetry/positive-definiteness tags and no CG option. The right-hand side is
one at transient array index zero. `Q` is the interior restriction and is never
renormalized. For custom selection, visits stop at the first endpoint hit even
if the full transition would allow leaving endpoints. Finite visits require
eventual endpoint hitting.

`result.converged` requires a successful Lineax status, finite output, and a
fresh infinity-norm residual no larger than `atol+rtol`. Always check it.
`result.stats` preserves the installed Lineax solver's iteration counters; GMRES
`num_steps` is not a count of individual matvecs. Failures retain the candidate
iterate for diagnosis, but it must not be treated as a valid occupation vector.

A tiny internal residual establishes convergence for the **approximate operator**.
Near neutrality, long absorption times can amplify its error. Tests and small
benchmarks also evaluate residuals against the exact dense transition matrix,
occupation-vector error, and total absorption time. Large systems have sampled
one-step accuracy checks; large near-neutral solves are not certified by them.
There is currently no preconditioner or rigorous a posteriori exact-model error
bound. Derivative accuracy is measured separately from value accuracy; the optional
larger degree alone does not certify gradients of ill-conditioned solves.
