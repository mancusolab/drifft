# drifft

Matrix-free Wright–Fisher transition operators in JAX, with an exact local
diagonal–Toeplitz–diagonal binomial factorization and Lineax occupation solves.
The statistical model uses `M=2N` gene copies and symmetric underdominance
`q_s(x) = (x-s*x*(1-x))/(1-2*s*x*(1-x))`, for `0 <= s <= 1`.

## Installation and use

```sh
uv sync
uv run pre-commit install
uv run pytest -p no:capture
```

```python
import jax
import jax.numpy as jnp
import drifft as wf

jax.config.update("jax_enable_x64", True)  # required; no import-time config changes
plan = wf.build_transition_plan(gene_copies=20000)
expect_next = jax.jit(wf.expected_next_values)
next_values = expect_next(plan, jnp.asarray(0.05), jnp.ones(20001))

segregating_values = jax.jit(wf.expected_segregating_values)(plan, jnp.asarray(0.05), jnp.ones(19999))
operator = wf.build_segregating_operator(plan, jnp.asarray(0.05))
next_mass = jax.jit(lambda mass: operator.T.mv(mass))(jnp.ones(19999))

result = wf.expected_visits(plan, jnp.asarray(0.05))
if not bool(result.converged):
    raise RuntimeError(f"{result.status}: residual={result.residual_norm}")
visits = result.visits  # index 0 is state i=1
```

Build the plan once and reuse it for different JAX scalar values of `selection_strength`.
Shapes/configuration are static PyTree metadata; array leaves remain dynamic.
`expected_visits` is already JIT-compiled. Forward functions are undecorated
so callers can choose a single JIT boundary around their surrounding computation.
Neither repeated matvecs nor solves use host callbacks, SciPy, or host transfers.

| API | Contract |
| --- | --- |
| `build_transition_plan(gene_copies, **options)` | Host-side planning, independent of selection |
| `expected_next_values(plan, selection_strength, values)` | Full operator, length `M+1`, absorbing endpoints |
| `expected_segregating_values(plan, selection_strength, values)` | Transient operator, length `M-1`, zero boundary padding |
| `build_segregating_operator(plan, selection_strength)` | `FunctionLinearOperator`; `.T` transposes the implemented map |
| `expected_visits(plan, selection_strength, ...)` | Solve `(I-Q.T)g=e1`; return `visits`, status, stats, residual and convergence flag |
| `apply_probabilities(plan, p, v)` | Evaluate at arbitrary 1-D success probabilities; extension point for another `q` |
| `apply_p_banded(M, s, v, tail_tol=...)` | Independent JAX banded reference; make `M` and tolerance static under JIT |

Inputs must be real vectors with the documented shapes; success probabilities
must lie in `[0,1]`, and selection must be a scalar in `[0,1]`. These value ranges
are caller preconditions. The production kernels do not insert runtime callbacks
to validate them. Plans reject invalid configuration and require float64.

The expectation functions map values assigned to next-generation states to their
expectations at each current state; they do not advance a probability distribution.
The transposed operator advances probability mass. Segregating expectations use
zero values at absorption and are not conditioned on survival. `expected_visits`
starts from one mutant copy and includes generation zero. The plan stores its
population size as `plan.gene_copies`; the result stores counts as `result.visits`.

## Approximation and configuration

Each integer center `j0` has local spacing `h=1/(M*p0*(1-p0))` in log-odds.
The FFT computes **binomial**, not diffusion/Gaussian, expectations at these nodes.
Convolutions are zero-padded to at least `R+2L-2`; the output slice begins at
`L-1`. Plans batch blocks with the same power-of-two FFT length and cache kernel
transforms. Disjoint core intervals determine routing; halos only support
interpolation and are never added as extra row contributions.

| Option | Default | Meaning |
| --- | --- | --- |
| `tail_tol` | `1e-12` | Discarded binomial mass budget at grid nodes and boundary rows; accepted range `[1e-15, .1)` |
| `interp_order` | `12` | Even polynomial stencil size, 2–16; 8 is faster in some configurations but less accurate |
| `min_mode_count` | `128` | Direct fallback when `M*min(p,1-p) <= cutoff`; minimum configurable value 8 |
| `block_spacing` | `1.0` | Center spacing in local standard deviations; `(0,4]` |
| `balance` | `True` | Apply a constant per-block shift to balance the diagonal factors |
| `dtype` | `jnp.float64` | Only float64 is supported in this release |

Input windows use host-side SciPy binomial quantiles rather than the suggested
Bernstein bound. This keeps factors better conditioned while retaining a
numerically checked tail budget. Stochastic ordering makes the two extreme
node probabilities sufficient to bound the two discarded tails throughout the
block. `plan.max_tail_mass` records the largest checked discarded mass. The
independent banded baseline uses the conservative Bernstein bound.

The modal probability uses Stirling corrections, avoiding subtraction of large
log-factorials. Input diagonal factors use the local recurrence; output factors
use a `log1p`/`expm1` softplus difference. A constant balancing shift preserves
the product. Intermediate host arithmetic uses `longdouble` where available,
but correctness does not rely on extra precision (macOS ARM uses float64 here).

The boundary sum has fixed width chosen by a binomial upper quantile at the
cutoff. Upper rows are reflected onto the lower boundary. Small problems with
`M <= 2*min_mode_count` use this direct path throughout. Batches of 64 rows bound
per-batch forward temporary storage and skip direct work for batches entirely
in the interior. Automatic transposition can retain `O(M*boundary_width)` residuals.
The boundary input gather occurs before the scan so its automatic transpose
accumulates a short boundary cotangent rather than a full vector per batch.

`tail_tol` is **not a bound on total operator error**. Interpolation and floating
point roundoff are separate errors; polynomial interpolation can introduce tiny
negative coefficients. Outputs are never clipped or renormalized. Consequently,
the operator is linear in signed inputs and its automatic transpose satisfies
the adjoint identity. Aggressive settings, especially a lower boundary cutoff
or wider blocks, require fresh accuracy checks. Conservative defaults target
normalized absolute matvec error below `1e-9`, not componentwise relative error.

For fixed configuration, block counts grow as `sqrt(M)`, their aggregate window
and grid lengths as `M`, and FFT work as approximately `M log M`. Bucket padding
has bounded overhead. The direct boundary path is at most linear in `M` for a
fixed cutoff and tolerance. The banded reference does `O(M**1.5)` work.
Planning/storage and measured runtime scaling are recorded in the benchmark
reports; these estimates do not imply a GPU speed advantage.

## Solves and limitations

The solver choices are `solver="gmres"` (default) and `"bicgstab"`. Defaults are
`rtol=1e-9`, `atol=1e-11`, `max_steps=2000`, `restart=50` (GMRES only). There are
no symmetry/positive-definiteness tags and no CG option. The right-hand side is
one at transient array index zero. `Q` is substochastic and never renormalized.

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
bound. Derivatives with respect to `s` are tested at an interior value against
finite differences, but no global smoothness guarantee is made at block seams,
the boundary switch, or endpoints of arbitrary supplied probability maps.

## Validation and benchmarks

```sh
uv run pytest -p no:capture                  # includes M=65536 sampled tests
uv run pytest -p no:capture -m 'not slow'    # omit large sampled-row cases
uv run ruff check .
uv run ruff format --check .
uv run ty check src tests benchmarks
uv run pre-commit run --all-files
uv build
uv run python benchmarks/benchmark.py --output benchmarks/cpu.json
uv run python benchmarks/benchmark.py --sizes 1024 10000 --selections 0 .5 \
  --orders 8 12 --tail-tols 1e-8 1e-12 --solve-max-m 0 \
  --output benchmarks/accuracy_sweep.json
```

The benchmark runs on the JAX-selected device. Use a GPU-enabled JAX installation
to repeat it there; no GPU results are claimed. It records package versions,
configuration, plan time/storage, compilation, synchronized steady-state forward
and transpose times, compiler temporary-memory estimates, process peak RSS,
sampled accuracy, and first/warm solve times for both solvers. Dense comparisons
are limited to `M <= 1024` by default. Run timing sweeps without competing jobs.

Use `--device cpu`, `--device gpu` or `--device tpu` to select the backend explicitly;
otherwise JAX selects it. Precision remains float64. `--quiet` suppresses progress
logging on stderr. JSON is written to stdout unless `--output` names a file.
Exit codes are 0 for a completed report, 2 for invalid arguments and 1 for runtime
or file errors. A completed report can contain an unconverged solve; inspect its
diagnostics. New reports carry `schema_version: 1`; the original measurements
stored alongside this script predate that field.

## Development checks

The pre-commit configuration is adapted from `jaxqtl`: it retains the standard
file checks and runs Ruff lint/format plus ty over `src`, `tests` and `benchmarks`.
The reference symlink and original task document are excluded. Install the hook
with `uv run pre-commit install`; CI runs all hooks, the full test suite and builds
using the locked uv environment on Python 3.11 and 3.12. The workflow follows the
[uv GitHub Actions guide](https://docs.astral.sh/uv/guides/integration/github/).

Public numerical docstrings describe shapes, approximation and failure contracts.
Keep numerical array fields dynamic and shape/configuration metadata static;
construction checks must inspect shapes without reading traced values. Use the
plan factory for validated input configuration. This package is unreleased;
rename APIs and their call sites directly without compatibility wrappers. Numerical changes require the independent reference and
transform tests; formatting and documentation changes do not need new tests.

The engineering checks retain this repository's ty toolchain. GPU/float32 and
global derivative guarantees remain outside the validated contract. Runtime
probability ranges remain caller preconditions to keep the original callback-free
kernel contract. There is no plan checkpoint format, sampling API or dataframe
adapter to validate; those checklist items do not apply to this package.

The direct/banded method can be preferable when plan reuse is low, a platform
handles gather/PMF arithmetic better than many small FFTs, or polynomial
interpolation is unsuitable. The provided baseline recomputes log-PMFs on each
call; it does not cache selection-specific band weights. Dense matvecs are much
faster for small systems once their matrix is built, at quadratic storage cost.
See [benchmarks/RESULTS.md](benchmarks/RESULTS.md) for measured tradeoffs.

API references checked during implementation:
[Lineax operators](https://docs.kidger.site/lineax/api/operators/),
[Lineax solvers](https://docs.kidger.site/lineax/api/solvers/), and
[linear_solve](https://docs.kidger.site/lineax/api/linear_solve/).
