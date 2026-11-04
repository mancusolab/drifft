# Choosing numerical algorithms

Choose a **backend** to represent one generation and an **evolution method**
to apply it across an interval. Start with FFT and Sequential; consider
alternatives when time or memory becomes a limitation.

## Backends

Backends prepare grids and weights for fixed population sizes and numerical
settings. Reuse them while changing selection, mutation, or the initial
distribution.

| Backend | When to use it |
| --- | --- |
| `FFT` | Default for one-generation transitions and Sequential evolution. |
| `NUFFT` | When retained FFT geometry is too large; trades storage for computation. |
| `Jacobi` | For reduced-state evolution with Krylov across long constant-size intervals. |
| `Dense` | As an independent binomial reference for small populations. |

For one generation, pass a backend instance to `Transition`:

```python
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)
selection = drifft.AdditiveSelection(s=0.02)
initial = jnp.zeros(129).at[1].set(1.0)  # 128 gene copies, 64 diploid individuals.
backend = drifft.FFT(diploid_size=64)
transition = drifft.Transition(backend, frequency_map=selection)
after_one_generation = transition.advance(initial)
```

Replace `drifft.FFT` with `drifft.NUFFT` to change the representation without
changing the selection model or state layout. Backends and `Demography` both
accept diploid population sizes. A size of $N$ gives $2N+1$ allele-count states.

When migrating older code, replace `gene_copies=M` with `diploid_size=M // 2`
and `source_gene_copies=M_s` with `source_diploid_size=M_s // 2`; both old
counts must be even. Positional size arguments also now count diploid individuals.
For example, the former `FFT(128)` becomes `FFT(64)`. Custom demographic backend
factories now receive $N$, rather than $2N$, and custom backends expose
`diploid_size`. `Demography(ne=...)` values are unchanged. Hypergeometric sampling
continues to accept population and sample copy counts, since it also supports
samples with an odd number of copies.

!!! warning "Jacobi reduced Krylov restrictions"

    All backends support arbitrary valid frequency maps for one-generation
    operations and Sequential evolution. Jacobi's reduced Krylov assembly supports
    a restricted family of selection and mutation maps within configured parameter
    bounds; check the [Jacobi API](../api/backends.md#drifft.Jacobi) before using it.

## Evolution methods

Evolution methods control repeated application of the represented transition.
They are configured separately from the backend:

| Method | When to use it |
| --- | --- |
| `Sequential` | Default: applies the transition each generation. Required for `model.expect()` and higher-order derivatives. |
| `PolynomialKrylov` | To approximate long intervals using operator applications, without shifted linear solves. Works with FFT or NUFFT for custom frequency maps. |
| `RationalKrylov` | To approximate long intervals using shifted solves. Pair with Jacobi to perform these solves in reduced coordinates. |

For a population history, pass a backend **factory** and a method instance.
Continuing the example above:

```python
history = drifft.Demography(ne=[64, 32, 64], generations=[100, 100])
model = drifft.DemographicTransition(
    history,
    frequency_map=selection,
    backend_factory=drifft.FFT,
    method=drifft.Sequential(),
)
result = model.advance(initial)
```

To use reduced-state evolution within intervals and FFT at size changes:

```python
reduced_model = drifft.DemographicTransition(
    history,
    frequency_map=selection,
    backend_factory=drifft.Jacobi,
    boundary_backend_factory=drifft.FFT,
    method=drifft.RationalKrylov(dimension=40),
)
reduced_result = reduced_model.advance(initial)
```

`dimension` limits the Krylov basis size; `rtol` and `atol` control its
power-approximation acceptance criterion. These settings do not control
backend error. Krylov methods support first-order sensitivities, but their
input-dependent projections do not provide the matched transpose required by
`model.expect()`.

RationalKrylov also accepts other backends, but its shifted solves can be
expensive in population coordinates. Neither Krylov method guarantees a speedup
or falls back to Sequential on failure. See the [method API](../api/methods.md)
for controls and the [Schiffels–Durbin example](../examples/schiffels_durbin.md)
for a larger Jacobi/RationalKrylov calculation.

## Compare configurations

Validate the final statistic and its gradients with the [accuracy checks](numerics.md).
Measure [setup, compilation, and warm execution](numerics.md#memory-and-timing)
separately. The [computational overview](../methods/overview.md) explains how
backend and evolution-method approximations combine.
