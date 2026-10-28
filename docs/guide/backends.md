# Choosing numerical algorithms

Choose a backend to represent the transition and an evolution method to apply
it over an interval. Begin with FFT and Sequential as a reference, then compare
alternatives at matched accuracy on the quantity you need.

```python
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)
history = drifft.Demography(ne=[32, 32], generations=[20])
model = drifft.DemographicTransition(
    history,
    backend=drifft.FFT,
    method=drifft.Sequential(),
    frequency_map=drifft.Underdominance(strength=0.01),
)
initial = jnp.zeros(65).at[1].set(1.0)
result = model.advance(initial)
```

This computes the recurrence generation by generation. The following choices
change its numerical representation or the way repeated evolution is computed.

## Match the backend to the model

| Backend | Representation | Useful when | Model support |
| --- | --- | --- | --- |
| FFT | Local convolutions and interpolation | Repeated one-generation operations | Built-in and custom maps |
| NUFFT | Binomial samples and polynomial evaluation | Comparing memory and accuracy tradeoffs with FFT | Built-in and custom maps |
| Jacobi | Reduced polynomial state | Long constant-size portions of a history | Neutral and symmetric underdominance |

FFT caches more geometry. NUFFT regenerates binomial weights in batches,
trading retained storage for computation. Both support `Transition`, including
expectations, size changes, and expected visits. Neither is universally faster;
measure preparation, compilation, and warm execution separately.

Jacobi uses an explicit FFT or NUFFT `boundary_backend` for size changes.
Its reduced matrix is dense: storage and matvec cost grow as $O(r^2)$ in rank
$r$, and LU factorization costs $O(r^3)$. Speedups over long intervals come
from working in this reduced space and reducing the number of evolution steps.

## Match the method to the interval

| Method | Backends | Work performed | Expectations |
| --- | --- | --- | --- |
| Sequential | FFT, NUFFT | One transition application per generation | Supported |
| PolynomialKrylov | FFT, NUFFT, Jacobi | Operator applications to construct a smaller power approximation | Forward propagation only |
| RationalKrylov | FFT, NUFFT, Jacobi | Shifted solves to construct a smaller power approximation | Forward propagation only |

Polynomial Krylov avoids shifted linear solves. Its basis storage grows with
working-state size times `dimension`; long intervals may need a larger basis.
Neither Krylov method automatically restarts or falls back to Sequential.
Rational Krylov with Jacobi reuses a dense LU factorization during basis
construction. With FFT or NUFFT it instead uses iterative, unpreconditioned
GMRES solves, each of which can require many transition applications.

!!! tip "Amortize setup over the work you actually need"

    For a short interval, Sequential may beat the setup cost of a Krylov method.
    For long intervals, compare total time including backend preparation,
    factorization, and decoding. Increase rank and Krylov dimension independently
    when checking accuracy.

For a supported neutral or underdominant model, try:

```python
accelerated = drifft.DemographicTransition(
    history,
    backend=drifft.Jacobi,
    boundary_backend=drifft.FFT,
    method=drifft.RationalKrylov(dimension=32),
    frequency_map=drifft.Underdominance(strength=0.01),
)
accelerated_result = accelerated.advance(initial)
```

Both examples use the same biological model and input. Compare their outputs
at tighter settings before comparing execution time.

!!! warning "Check supported combinations"

    Jacobi currently accepts only `Neutral` and `Underdominance`, including when
    another map has equivalent parameter values. Both Krylov methods provide
    forward propagation only; use Sequential when you need `model.expect()`.

## Representation and accuracy choices

NUFFT supports ordinary Lobatto interpolation and `encoding="filtered"`.
Filtering can reduce the number of samples needed by its spectral certificate;
it does not guarantee faster execution or improved derivative accuracy.
Jacobi's `rank` controls polynomial compression, while a Krylov method's
`dimension` controls its power approximation. These are separate error sources.

Use [accuracy and differentiation](numerics.md) to choose and check settings.
[Population histories](demography.md) explains boundary timing and factories.
The [backend API](../api/backends.md) and [method API](../api/methods.md) list
constructor options. For the numerical mechanisms, see [FFT convolution](../methods/fft.md),
[Chebyshev/NUFFT transforms](../methods/nufft.md), and
[Jacobi reduction and Krylov powers](../methods/jacobi_krylov.md).
