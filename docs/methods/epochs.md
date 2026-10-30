# Direct epoch operations

The [population-history guide](../guide/demography.md) covers the usual workflow.
For experiments or custom orchestration, an evolution method can also act
directly on a prepared backend, including arbitrary immigration vectors. This
page is an application recipe; the [Krylov chapter](jacobi_krylov.md) explains
the power algorithms and the [Jacobi chapter](jacobi.md) explains its backend.

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
A `Demography([64, 64], [250])` interval instead uses 249 hold generations and one FFT boundary generation, following
the history's interval convention even when the sizes are equal. They use
different approximations to the same biological recurrence and can produce
different results.

## Mass and immigration

Full-state input has `gene_copies + 1` entries, including count zero and fixation.
With `segregating=True`, input has `gene_copies - 1` entries and each generation
discards boundary mass. `immigration` adds the supplied vector **after every
generation**, including the last. Zero generations returns the original mass
without immigration. Signed vectors are allowed; outputs are never clipped or
renormalized.

!!! note "Mutation models"

    Singleton immigration is an infinite-sites source of new segregating mass.
    It does not apply recurrent mutation or backmutation to existing alleles.
    Recurrent mutation is a `Mutation` frequency map, optionally composed with
    one diploid selection rule in either order. Jacobi supports these maps;
    full-state evolution lets endpoint mass return to the interior, while
    `segregating=True` still discards endpoint mass after every generation.

## Accuracy and failures

Jacobi's `rank` controls polynomial truncation. Its default uses an ideal
one-step tail bound. `max_strength` controls assembly padding and is checked at
runtime against the absolute values of both diploid coefficients; choose it
when preparing geometry for the parameter range you need. It must remain
strictly below one. Neither setting certifies the final demographic result.

Both Krylov methods use `dimension` to control the power approximation. The reported
absolute `error_estimate` measures differences in the output distribution using
the sum of absolute values. It combines a comparison with a smaller Krylov basis
and a check of how closely the reduced trajectory follows the recurrence.

!!! warning "Estimate, not certificate"

    These diagnostics assess the power approximation, not errors in the backend
    representation itself. `converged=True` does not certify total accuracy.
    Validate a new regime against FFT/NUFFT with Sequential, and increase Jacobi
    rank and Krylov dimension independently.

By default, a nonfinite output or an estimate above
`atol + rtol * sum(abs(distribution))` raises a runtime error. For direct method
calls, use `throw=False`
to inspect `result.converged` and `result.error_estimate`. Demographic calls
report these estimates in `result.diagnostics`. Domain errors still
raise, and a failure under `vmap` may fail the whole batch. Increase the Krylov
dimension or use FFT/NUFFT with Sequential after nonconvergence. A sequential
FFT/NUFFT reference also checks errors in the Jacobi reduction that the power
estimate does not cover.

## Differentiation and demographic integration

Selection, mutation rates, initial mass, and immigration are dynamic arrays;
dimensions and generation counts are static. Krylov methods use custom
first-order power-sensitivity rules, including a low-rank operator pullback.
They support zero and invariant seeds when derivative convergence checks pass.
Sensitivity or pullback failures raise even with `throw=False`. Higher-order
operator sensitivities are not yet implemented. See the
[autodiff chapter](autodiff.md#accuracy-and-limitations) for the accuracy contract.
Preparation, reconstruction, and demographic boundaries remain differentiated
through JAX. Numerical error estimates are excluded from differentiation.

Because their bases depend on the input, neither Krylov method provides a
matched expectation transpose. Use Sequential when you need `model.expect()`.

```python
history = drifft.Demography(ne=[64, 40, 80], generations=[250, 100])
model = drifft.DemographicTransition(
    history,
    backend=drifft.Jacobi,
    boundary_backend=drifft.FFT,
    method=drifft.RationalKrylov(),
    frequency_map=drifft.Neutral(),
)
final = model.advance_segregating(initial, mutation_rate=1e-8).distribution
```

Constant-size portions use the selected backend and method; the final, possibly
rectangular step of each interval uses FFT/NUFFT. Immigration on that step uses the
destination gene-copy count. `model.expect()` is available only with Sequential
epochs. Equilibrium initialization remains explicit: use
`Transition.expected_visits()` and scale its visits by the singleton input.
