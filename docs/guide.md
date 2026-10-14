# Usage and numerical guide

[Back to the README](../README.md)

Matrix-free Wright–Fisher transitions with interchangeable FFT and Chebyshev/NUFFT
backends in JAX, with Lineax expected-visits solves.
The statistical model uses `M=2N` gene copies and accepts arbitrary JAX-compatible
selection functions. The built-in `Underdominance` model uses
`q_s(x) = (x-s*x*(1-x))/(1-2*s*x*(1-x))`, for `0 <= s <= 1`.

## Installation and use

Requires Python 3.12 or newer.

```sh
uv sync
uv run pre-commit install
uv run pytest -p no:capture
```

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)
backend = drifft.FFT(gene_copies=20000)  # or drifft.NUFFT(gene_copies=20000)
transition = drifft.Transition(
    backend, selection=drifft.Underdominance(strength=0.05)
)

# Compile stable functions that accept the transition as a PyTree argument.
expect = eqx.filter_jit(drifft.Transition.expect)
advance = eqx.filter_jit(drifft.Transition.advance)
expected_values = expect(transition, jnp.ones(20001))
next_distribution = advance(transition, jnp.zeros(20001).at[1].set(1.0))

result = transition.expected_visits()  # already JIT-compiled
if not bool(result.converged):
    raise RuntimeError(f"{result.status}: residual={result.residual_norm}")
visits = result.visits  # index zero is one mutant copy; includes generation zero
```

Build the backend once and reuse it across transitions. `Transition` binds the backend
without rebuilding it. Selection parameters are dynamic array leaves, while
shape/configuration metadata are static. For another selection strength, construct
`Transition(backend, Underdominance(new_strength))` and pass it to the same compiled
function. Scalar strengths are normalized to float64 automatically.

| API | Contract |
| --- | --- |
| `FFT(gene_copies, **options)` / `NUFFT(gene_copies, **options)` | Prepare destination geometry outside JIT |
| `AbstractTransitionBackend` | Equinox abstract base; implement linear `expect(probabilities, values)` |
| `Transition(backend, selection=...)` | Bind a JAX-compatible frequency-to-probability callable |
| `Transition.from_probabilities(backend, probabilities)` | Bind explicit probabilities, one per source row |
| `transition.expect(values)` | Destination values to source expectations: `P @ values` |
| `transition.advance(distribution)` | Source mass to destination mass: `P.T @ distribution` |
| `transition.expect_segregating(values)` | Interior expectations with endpoint values zero |
| `transition.advance_segregating(distribution)` | Interior mass propagation, discarding endpoint hits |
| `transition.as_lineax(segregating=False)` | Expectation operator; `.T` advances mass |
| `transition.expected_visits(**options)` | Square-transition visits from one copy, with convergence diagnostics |

`expect` and `advance` have different directions and are intentionally named
explicitly. Neither changes the input, clips outputs, nor normalizes probability
mass. Segregating operations remove endpoint contributions; they do not condition
on survival. Repeated operations remain JIT-compatible; NUFFTs use FINUFFT’s compiled FFI
rather than Python host callbacks.

Custom selection models use the same interface:

```python
# Any JAX-compatible callable that returns one probability per frequency works.
custom = drifft.Transition(backend, selection=lambda frequency: frequency**2)
# An Equinox callable can carry dynamic parameters for inference or sweeps.
# Alternatively, supply already evaluated probabilities explicitly.
offspring_probabilities = custom.probabilities
explicit = drifft.Transition.from_probabilities(backend, offspring_probabilities)
```

Model probabilities are evaluated during application, keeping parameter updates
and differentiation connected. With explicit probabilities, construct them inside
the differentiated function if gradients through their construction are needed.
Use Equinox fields for trainable parameters instead of closing over changing
Python values in fresh lambdas.

For population-size changes, the backend specifies destination size and
`Transition(backend, selection=..., source_gene_copies=...)` specifies source size.
`expect` maps destination-length values to source-length expectations; `advance`
performs the reverse mapping for mass. Explicit probabilities may also describe
sampled source rows. Segregating and expected-visits operations assume complete,
ordered allele-count grids; expected visits additionally requires equal source
and destination sizes. A full changing-size demographic workflow is not provided.

Inputs must have the documented shapes. Frequencies and probabilities must be in
`[0, 1]`, and `Underdominance.strength` must be scalar and in `[0, 1]`. Shape checks
raise during construction or tracing; finite values and numerical ranges remain
caller preconditions to avoid runtime callbacks. Backends require float64.

## Backend selection

Both backends implement `AbstractTransitionBackend`, an abstract Equinox module.
`Transition` owns selection, source size, propagation, and solver integration;
backends own selection-independent geometry and the expectation approximation.
Choose a backend explicitly and reuse it across selection strengths.

`FFT` uses local log-odds blocks, cached Fourier coefficients, and interpolation.
It favors repeated applications at a fixed size, at the cost of larger geometry
and XLA temporary storage. Its constructor accepts `tail_tol`, `interp_order`,
`min_mode_count`, `block_spacing`, and `balance`. Host NumPy/SciPy preparation
uses binomial quantiles and extended precision for conditioned Fourier weights;
application and differentiation use JAX arrays. Interpolation boundaries can
limit gradient smoothness and should be checked for inference workloads.

`NUFFT` stores a small node vector and regenerates sampling weights, trading
extra work per application for lower stored geometry and XLA temporary storage.
Its polynomial derivative rules remain finite at probability endpoints.
These custom operations use experimental `jax.experimental.hijax.HiPrim` to
preserve explicit JVP, transpose, and batching behavior under JAX transforms.
Hijax does not itself make gradients faster or reduce demographic recompilation:
changing population shapes can require new compilations for either backend.

## NUFFT approximation and configuration

`expect` samples the Bernstein polynomial at fixed Chebyshev–Lobatto nodes,
computes cosine coefficients using a real-FFT DCT-I, then evaluates them at the
selection probabilities using FINUFFT. The degree and sample geometry depend on
the destination population and accuracy settings, never on the input vector or
selection strength. The backend stores only one node vector.

| Option | Default | Meaning |
| --- | --- | --- |
| `accuracy` | `1e-10` | Nominal value-error budget; `[1e-13, .1)` |
| `tail_tol` | allocated automatically | Optional stricter omitted binomial mass per node |
| `nufft_tol` | `min(1e-13, accuracy/4)` | Requested FINUFFT tolerance; at least `1e-15` |
| `batch_size` | `32` | Sampling nodes processed together |
| `derivative_accuracy` | `False` | Increase degree by 50%, capped at exact degree; heuristic, not a derivative certificate |
| `dtype` | `jnp.float64` | Only float64; enable x64 before planning |

The nominal budget is split equally between spectral interpolation, sampling,
NUFFT evaluation and roundoff. The fixed degree uses the value bound
`4*M/L * exp(-L**2/(2*M))`; at `L=M` interpolation is exact in exact arithmetic.
Per-node tail error is divided by twice a conservative Lobatto Lebesgue constant
`1 + (2/pi)*log(L+1)`. Plan fields expose both the spectral bound and the
amplified sampling bound. NUFFT tolerance and the roundoff allocation are
**not rigorous error certificates**; both require empirical validation.

Sampling builds nonnegative weights from binomial mode-relative ratios, then
normalizes each retained stencil. A Bernstein tail bound controls discarded
mass. This changes a sampling-node expectation by at most twice that mass times
`max(abs(values))`. It is fixed linear preprocessing, not normalization of a
transition output or segregating row. Endpoint samples and transition rows are
exact. Spectral interpolation can still introduce tiny negative coefficients;
outputs are neither clipped nor renormalized.

`advance` is the algebraic transpose of the same approximation. Its sampling
stage regenerates each stencil batch and scatters into one output accumulator.
There is no permanent node-by-stencil weight table. Expected work is
`O(L*W + L*log(L) + source_gene_copies)` up to NUFFT tolerance factors, with
`L` and `W` approximately proportional to `sqrt(M*log(M))`. Working array storage
is `O(M + source_gene_copies + L + batch_size*W)`, excluding library-managed
FINUFFT scratch. XLA memory reports do not measure that external scratch.

Selection derivatives evaluate the derivative polynomial in coefficient space,
including at probability zero and one. The transpose has a matched derivative
rule; it does not differentiate the singular angular coordinate. First and
second polynomial derivatives, propagation gradients and small expected-visits
gradients have independent reference tests. Sampling nodes are fixed geometry:
differentiating the backend's node leaves is rejected. Differentiate selection
parameters or input vectors while holding the backend fixed.

### Dependency and hardware constraints

`jax-finufft` remains a core dependency for the NUFFT implementation. The dependency follows upstream `jax-finufft` `main`; `uv.lock` records the exact
commit. This includes batching fixes absent from release 1.3.1 and permits the
latest JAX on supported Python versions. Installing from Git builds the native
extension and requires a C++ toolchain. The build uses installed FFTW development
libraries when available; otherwise it downloads and builds FFTW automatically
(OpenMP is optional). Ubuntu CI installs `libfftw3-dev` to avoid that extra build.
CPU execution is tested on Apple Silicon. CUDA requires a
compatible CUDA-enabled FINUFFT installation and has not been validated here;
TPU is unsupported. No dense fallback is used when a device lacks NUFFT support.

## Solves and limitations

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

## Validation and benchmarks

```sh
uv run pytest -p no:capture
uv run ruff check src tests benchmarks
uv run ruff format --check src tests benchmarks
uv run ty check src tests benchmarks
uv run pre-commit run --all-files
uv build
uv run python benchmarks/benchmark.py --device cpu --output benchmarks/bernstein_cpu.json
uv run --group fastdtwf python benchmarks/compare_fastdtwf.py --sizes 64 256 1024
```

The spectral benchmark reports compilation, warm execution, sampled forward and
selection-JVP accuracy, backend storage, XLA temporary storage, mass defects and
small-system solve diagnostics. Use `--source-gene-copies` for rectangular
transitions, and `--derivative-accuracy` to evaluate the larger-degree policy.
Multiple RHS use `jax.vmap`; this processes independent vectors rather than
claiming a shared batched NUFFT speedup.

Use `benchmarks/compare_backends.py` to compare both installed backends.
`compare_startup.py` and `compare_epochs.py` also compare both against fastDTWF.

See [historical replacement results](../benchmarks/BERNSTEIN.md) for measured limitations.
The older [CPU report](../benchmarks/RESULTS.md) and
[fastDTWF comparison](../benchmarks/FASTDTWF.md) describe the previous backend and
record earlier revisions; their measurements have not been regenerated for this refactor.
