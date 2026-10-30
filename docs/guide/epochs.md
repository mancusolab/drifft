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
