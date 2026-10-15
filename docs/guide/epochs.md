# Constant-size epochs

An epoch holds population size and the frequency map fixed for several
generations. `Sequential` repeats FFT or NUFFT transitions. `RationalKrylov`
uses a reduced Jacobi operator to approximate the whole epoch.

```python
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)

epoch = drifft.RationalKrylov(128, krylov_dimension=40)
initial = jnp.zeros(127).at[0].set(1.0)
source = jnp.zeros_like(initial).at[0].set(128 * 1e-8)
result = epoch.advance(
    initial,
    frequency_map=drifft.Neutral(),
    generations=250,
    segregating=True,
    immigration=source,
)
final = result.distribution
```

Replace the constructor with `drifft.Sequential(drifft.FFT(128))` to evaluate
the same recurrence generation by generation. Both implement
`AbstractEpochBackend`. Geometry is prepared outside JIT and reused across
selection strengths; `.advance()` compiles automatically.

## Mass and immigration

Full-state input has `gene_copies + 1` entries, including absorbing endpoints.
With `segregating=True`, input has `gene_copies - 1` entries and each generation
discards boundary mass. `immigration` adds the supplied vector **after every
generation**, including the last. Zero generations returns the original mass
without immigration. Signed vectors are allowed; outputs are never clipped or
renormalized.

!!! note "Mutation models"

    Singleton immigration is an infinite-sites source of new segregating mass.
    It does not apply recurrent mutation or backmutation to existing alleles.
    RationalKrylov currently accepts only `Neutral` and `Underdominance`.
    Use Sequential with a composed frequency map for recurrent mutation.

## Reduced operator and power action

Let $P$ be the one-generation expectation matrix on allele counts. The Jacobi
approximation factors it as $P\approx DE$, where $E$ encodes an observable in
polynomial coefficients and $D$ evaluates those coefficients after selection.
The full-state representation includes two endpoint coordinates. The
segregating representation retains only the interior polynomial coefficients.

With $H=ED$, forward evolution through $g\geq1$ generations is

$$p_g\approx E^T(H^T)^{g-1}D^T p_0.$$

For constant immigration $b$, initialize $y_0=D^Tp_0$ and apply

$$
y_{t+1}=H^Ty_t+D^Tb,\qquad
p_g\approx E^Ty_{g-1}+b.
$$

The implementation turns this affine recurrence into a homogeneous augmented
matrix. For its matrix $A$ and exponent $k=g-1$, rational Arnoldi builds an
orthonormal basis from $(I+\tau(I-A))^{-1}$ with $\tau=k/8$ and two rounds of
reorthogonalization. It projects the **original** matrix, $B=V^TAV$, then
evaluates $VB^kV^Ty_0$ by binary powering the small matrix.

The Jacobi assembly uses parity-separated tridiagonal solves and polynomial
recurrences. The reduced matrix and its LU factors are dense: storage is
$O(r^2)$ and factorization is $O(r^3)$ for polynomial rank $r$. Grid encoding
streams polynomials using $O(M+r)$ working storage and $O(Mr)$ work.
Quadrature and binomial-stencil geometry are also retained. Only half the
Gauss–Jacobi nodes are stored: paired nodes have equal weights and polynomial
values related by $p_n(-z)=(-1)^n p_n(z)$. Even and odd polynomial contributions
recover both halves for arbitrary distributions. A central node, when present,
has half weight in each reflected contribution. Stencil indices are generated
from one int32 starting index per row. These storage changes preserve the
polynomial rank, quadrature order, float64 weights, and truncation tolerance.
This backend does not use HODLR compression.

## Accuracy and failures

`rank` controls polynomial truncation. Its default uses an ideal one-step tail
bound. `max_strength` controls assembly padding and is checked at runtime;
raise it when preparing geometry for stronger selection. Neither setting
certifies the final demographic result.

`krylov_dimension` controls the power approximation. The reported absolute L1
`error_estimate` takes the maximum of two diagnostics:

- The difference between the full Krylov space and its leading half. A complete
  reduced basis does not need this comparison.
- Accumulated decoded recurrence defects $R=AV-VB$, weighted along the
  projected trajectory. This catches cases where both nested approximations
  decay prematurely. Accumulation costs $O(gm^2)$ for Krylov dimension $m$.

!!! warning "Estimate, not certificate"

    The defect estimate uses contractivity of the ideal population transition;
    the approximate reduction and codec need not satisfy that property exactly.
    These diagnostics exclude polynomial truncation, quadrature, assembly, and
    floating-point errors. `converged=True` does not certify total accuracy.
    Validate a new regime against Sequential and increased rank/dimension.

By default, a nonfinite output or an estimate above
`atol + rtol * sum(abs(distribution))` raises a runtime error. Use `throw=False`
to inspect `result.converged` and `result.error_estimate`. Domain errors still
raise, and a failure under `vmap` may fail the whole batch. Increase the Krylov
dimension or use Sequential after nonconvergence.

## Differentiation and demographic integration

JAX differentiates the assembled matrix and the input-dependent Krylov basis.
Selection, initial mass, and immigration are dynamic arrays; dimensions and
generation counts are static. Derivatives at zero seeds or changes in Arnoldi
breakdown are not guaranteed. Reverse-mode encoding may retain $O(Mr)$ state.
The diagnostic defect accumulation is excluded from differentiation.

Because its basis depends on the input, RationalKrylov is not a fixed linear
operator with a matched transpose. It does not implement
`AbstractTransitionBackend.expect` or a Lineax operator.

```python
history = drifft.Demography(ne=[64, 40, 80], generations=[250, 100])
model = drifft.DemographicTransition(
    history,
    backend=drifft.FFT,
    epoch_backend=drifft.RationalKrylov,
    frequency_map=drifft.Neutral(),
)
final = model.advance_segregating(initial, mutation_rate=1e-8)
```

Constant-size portions use the epoch backend; the final, possibly rectangular
step of each interval uses FFT/NUFFT. Immigration on that step uses the
destination gene-copy count. `model.expect()` is available only with Sequential
epochs. Equilibrium initialization remains explicit: use
`Transition.expected_visits()` and scale its visits by the singleton input.
