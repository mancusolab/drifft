# Constant-size methods

An epoch holds population size and the frequency map fixed for several
generations. A backend supplies the numerical representation; a method computes
the multi-generation evolution. `Sequential()` repeats FFT or NUFFT transitions.
`PolynomialKrylov()` uses operator applications from any backend.
`RationalKrylov()` builds its basis using shifted linear solves: dense LU for
Jacobi, or matrix-free GMRES for FFT/NUFFT. Compose them through `DemographicTransition`:

```python
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)

model = drifft.DemographicTransition(
    drifft.Demography(ne=[64, 64], generations=[250]),
    backend=drifft.Jacobi,
    boundary_backend=drifft.FFT,
    method=drifft.RationalKrylov(dimension=40),
    frequency_map=drifft.Neutral(),
)
initial = jnp.zeros(127).at[0].set(1.0)
result = model.advance_segregating(
    initial,
    mutation_rate=1e-8,
)
final = result.distribution
```

Use `backend=drifft.FFT` and `method=drifft.Sequential()` to evaluate the
recurrence generation by generation. Methods implement `AbstractEpochMethod`
and contain no population geometry. Geometry is prepared outside JIT and reused
across selection strengths; numerical methods compile automatically.

## Choosing a method

| Method | Backends | Computation |
| --- | --- | --- |
| `Sequential` | FFT, NUFFT | Apply one generation at a time |
| `PolynomialKrylov` | FFT, NUFFT, Jacobi | Build a polynomial basis using operator applications |
| `RationalKrylov` | FFT, NUFFT, Jacobi | Build a basis using shifted linear solves |

For example, use polynomial Krylov with FFT geometry:

```python
polynomial_model = drifft.DemographicTransition(
    drifft.Demography(ne=[8, 8], generations=[30]),
    backend=drifft.FFT,
    method=drifft.PolynomialKrylov(dimension=32),
    frequency_map=drifft.Underdominance(0.02),
)
polynomial_result = polynomial_model.advance(jnp.zeros(17).at[1].set(1.0))
```

Polynomial Krylov builds a basis from the initial state and repeated operator
applications, then powers the projected matrix. It needs no shifted solves and
never constructs a dense population transition. Its basis requires storage
proportional to working-state size times `dimension`. Long intervals may need a
larger basis; no automatic restart or Sequential fallback is performed.

To try rational Krylov with the same FFT model, set
`method=drifft.RationalKrylov(dimension=64)`. FFT and NUFFT use unpreconditioned
Lineax GMRES internally. Each inner solve has its own iteration limit and
residual check; failure produces `status=EvolutionStatus.linear_solve_failed`.
Increasing the outer basis dimension does not fix an unsuccessful inner solve.
Jacobi retains its reusable LU factorization. Solver selection is internal.


## Direct method calls

For numerical experiments or custom history orchestration, a method can also
advance a prepared backend directly, including arbitrary immigration vectors:

```python
backend = drifft.Jacobi(128)
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

This applies the hold backend for all 250 generations. The demographic example
instead uses 249 hold generations and one FFT boundary generation, following
the history's interval convention even when the sizes are equal. They use
different approximations to the same biological recurrence and can produce
different results.

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
    Jacobi currently accepts only `Neutral` and `Underdominance`.
    Use FFT or NUFFT with Sequential and a composed frequency map for
    recurrent mutation.

## Accuracy and failures

Jacobi's `rank` controls polynomial truncation. Its default uses an ideal
one-step tail bound. `max_strength` controls assembly padding and is checked at
runtime; raise it when preparing geometry for stronger selection. Neither
setting certifies the final demographic result.

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

JAX differentiates the operator applications and the input-dependent Krylov basis.
Selection, initial mass, and immigration are dynamic arrays; dimensions and
generation counts are static. Derivatives at zero seeds or changes in Arnoldi
breakdown are not guaranteed. Differentiation can require substantially more
memory than forward evaluation.
The accumulated recurrence-defect diagnostic is excluded from differentiation.
RationalKrylov differentiates GMRES solves implicitly through Lineax. A failed
tangent or transpose solve raises even when the primal call uses `throw=False`.

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

Jacobi assembles a dense reduced matrix without constructing the full
population transition. Storage grows quadratically with rank, and the internal
LU factorization grows cubically. RationalKrylov reuses that factorization
throughout basis construction. Increasing rank therefore has a different cost
from increasing Krylov dimension.

## Migration from the combined epoch backend

The experimental epoch API now separates geometry from algorithms. There is no
public `Epoch` wrapper.

| Earlier API | Current API |
| --- | --- |
| `RationalKrylov(n, rank=r, krylov_dimension=k)` | `Jacobi(n, rank=r)` and `RationalKrylov(dimension=k)` |
| `Sequential(backend)` | `Sequential()` with backend passed to `advance` |
| `epoch.advance(mass, ...)` | `method.advance(backend, mass, ...)` |
| `epoch_backend=RationalKrylov` | `backend=Jacobi, boundary_backend=FFT, method=RationalKrylov()` |
| `final = model.advance(initial)` | `final = model.advance(initial).distribution` |

Move `rank`, `max_strength`, and `one_step_tolerance` to Jacobi construction.
Keep `rtol` and `atol` on RationalKrylov. `AbstractEpochMethod` replaces
`AbstractEpochBackend`. Demographic forward methods return convergence status
and interval diagnostics alongside the distribution. Single-generation
`Transition` methods and expected-visits results retain their existing API.
