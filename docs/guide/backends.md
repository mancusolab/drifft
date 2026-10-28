# Choosing numerical algorithms

Start with FFT for one-generation calculations and FFT with Sequential for
population histories. These support every frequency map and both propagation
and expectations. Change that configuration when you have a specific reason:

| Your calculation | First choice | When to try an alternative |
| --- | --- | --- |
| One generation or expected visits | FFT | Compare NUFFT if retained geometry uses too much memory |
| A history with short intervals or general selection | FFT + Sequential | Compare NUFFT for memory; test Krylov if repeated stepping dominates runtime |
| Long intervals with neutral or symmetric underdominant selection | Compare Jacobi + RationalKrylov with FFT + Sequential | Keep the faster configuration at matched accuracy, including setup |
| Conditional expectations through a history | FFT or NUFFT + Sequential | Krylov methods provide forward propagation only |

These are starting points for measurement. The [accuracy guide](numerics.md)
explains how to check that two configurations solve your problem equally well.

## Accelerate a long interval

Jacobi compresses the working state, and RationalKrylov approximates repeated
evolution within it. Here is a complete example using neutral evolution:

```python
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)
history = drifft.Demography(ne=[32, 32], generations=[200])
initial = jnp.zeros(65).at[1].set(1.0)

reference = drifft.DemographicTransition(
    history,
    backend=drifft.FFT,
    method=drifft.Sequential(),
    frequency_map=drifft.Neutral(),
)
accelerated = drifft.DemographicTransition(
    history,
    backend=drifft.Jacobi,
    boundary_backend=drifft.FFT,
    method=drifft.RationalKrylov(dimension=32),
    frequency_map=drifft.Neutral(),
)
reference_result = reference.advance(initial)
accelerated_result = accelerated.advance(initial)
difference = jnp.sum(jnp.abs(
    reference_result.distribution - accelerated_result.distribution
))
print(f"Distribution L1 difference: {difference:.3e}")
```

The printed difference compares the full output distributions. FFT is an
approximate reference; repeat at tighter settings as described in
[accuracy and differentiation](numerics.md). This small population demonstrates
the workflow, not a timing advantage.

!!! tip "Measure the complete calculation"

    Include preparation, compilation, factorization, and decoding for a first-use
    measurement. Measure subsequent execution separately when you will reuse the
    model. Synchronize JAX outputs before recording elapsed time.

## When another choice helps

**NUFFT** retains less geometry and regenerates binomial weights in batches.
It trades storage for computation. Its ordinary and filtered encodings offer
different accuracy and sample-count tradeoffs; compare them on your workload.

**PolynomialKrylov** builds a power approximation using operator applications,
avoiding shifted linear solves. It works with FFT, NUFFT, or Jacobi. It can
require a larger basis than RationalKrylov for long intervals.

**RationalKrylov with FFT or NUFFT** uses iterative GMRES shifted solves.
Each basis vector can require many transition applications, so compare its
actual runtime with Sequential. With Jacobi, it reuses a dense LU factorization.
Neither Krylov method automatically restarts or falls back to Sequential.

!!! warning "Jacobi has a restricted selection model"

    Use `Neutral` or `Underdominance` with Jacobi. General diploid selection,
    mutation maps, and custom maps currently require FFT or NUFFT, even when
    their parameters happen to reproduce an accepted model. Jacobi also needs
    an explicit FFT or NUFFT `boundary_backend` for population-size changes.

## Understand the costs

Jacobi's dense rank-$r$ matrix costs $O(r^2)$ to store and apply, and $O(r^3)$
to factor. Compression does not make its matvec an FFT. The saving over a long
interval comes from working in a smaller state and reducing repeated steps;
encoding and decoding still cost work.

The [FFT](../methods/fft.md), [NUFFT](../methods/nufft.md), and
[Jacobi](../methods/jacobi.md), and [Krylov](../methods/jacobi_krylov.md) chapters explain the mechanisms.
The [backend](../api/backends.md) and [method](../api/methods.md) API pages list
configuration options; the [population-history guide](demography.md) explains
boundary timing and factories.
