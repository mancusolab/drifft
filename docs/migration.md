# Migrating the epoch API


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
