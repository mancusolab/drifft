# drifft

Matrix-free Wright–Fisher transition operators in JAX, with an exact local
diagonal–Toeplitz–diagonal binomial factorization and Lineax occupation solves.
The statistical model uses `M=2N` gene copies and accepts arbitrary JAX-compatible
selection functions. The built-in `Underdominance` model uses
`q_s(x) = (x-s*x*(1-x))/(1-2*s*x*(1-x))`, for `0 <= s <= 1`.

## Installation and use

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
plan = drifft.build_transition_plan(gene_copies=20000)
transition = drifft.Transition(
    plan, selection=drifft.Underdominance(strength=0.05)
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

Build the plan once and reuse it across transitions. `Transition` binds the plan
without rebuilding it. Selection parameters are dynamic array leaves, while
shape/configuration metadata are static. For another selection strength, construct
`Transition(plan, Underdominance(new_strength))` and pass it to the same compiled
function. Scalar strengths are normalized to float64 automatically.

| API | Contract |
| --- | --- |
| `build_transition_plan(gene_copies, **options)` | Host-side preparation for the destination population |
| `Transition(plan, selection=...)` | Bind a JAX-compatible frequency-to-probability callable |
| `Transition.from_probabilities(plan, probabilities)` | Bind explicit probabilities, one per source row |
| `transition.expect(values)` | Destination values to source expectations: `P @ values` |
| `transition.advance(distribution)` | Source mass to destination mass: `P.T @ distribution` |
| `transition.expect_segregating(values)` | Interior expectations with endpoint values zero |
| `transition.advance_segregating(distribution)` | Interior mass propagation, discarding endpoint hits |
| `transition.as_lineax(segregating=False)` | Expectation operator; `.T` advances mass |
| `transition.expected_visits(**options)` | Square-transition visits from one copy, with convergence diagnostics |

`expect` and `advance` have different directions and are intentionally named
explicitly. Neither changes the input, clips outputs, nor normalizes probability
mass. Segregating operations remove endpoint contributions; they do not condition
on survival. All repeated numerical operations are JAX-only with no host callbacks.

Custom selection models use the same interface:

```python
# Any JAX-compatible callable that returns one probability per frequency works.
custom = drifft.Transition(plan, selection=lambda frequency: frequency**2)
# An Equinox callable can carry dynamic parameters for inference or sweeps.
# Alternatively, supply already evaluated probabilities explicitly.
offspring_probabilities = custom.probabilities
explicit = drifft.Transition.from_probabilities(plan, offspring_probabilities)
```

Model probabilities are evaluated during application, keeping parameter updates
and differentiation connected. With explicit probabilities, construct them inside
the differentiated function if gradients through their construction are needed.
Use Equinox fields for trainable parameters instead of closing over changing
Python values in fresh lambdas.

For population-size changes, the plan specifies destination size and
`Transition(plan, selection=..., source_gene_copies=...)` specifies source size.
`expect` maps destination-length values to source-length expectations; `advance`
performs the reverse mapping for mass. Explicit probabilities may also describe
sampled source rows. Segregating and expected-visits operations assume complete,
ordered allele-count grids; expected visits additionally requires equal source
and destination sizes. A full changing-size demographic workflow is not provided.

Inputs must have the documented shapes. Frequencies and probabilities must be in
`[0, 1]`, and `Underdominance.strength` must be scalar and in `[0, 1]`. Shape checks
raise during construction or tracing; finite values and numerical ranges remain
caller preconditions to avoid runtime callbacks. Plans require float64.

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
bound. Derivatives with respect to selection strength are tested at an interior value against
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

## Comparison with fastDTWF

Install the optional, revision-pinned benchmark dependencies and run the CPU
comparison separately from the signed-vector benchmark:

```sh
uv run --group fastdtwf python benchmarks/compare_fastdtwf.py \
  --sizes 64 256 1024 4096 --repeats 3 --output benchmarks/fastdtwf_cpu.json
uv run --group fastdtwf pytest -p no:capture tests/test_fastdtwf_benchmark.py
```

This compares full distribution propagation and small-system expected visits
against [fastDTWF](https://github.com/jeffspence/fastDTWF). It records first-call
and warm timings, setup costs, approximation settings, and dense-reference errors.
To include Python startup, imports, plan construction, and first-call compilation,
run each backend in fresh processes at the pre-expansion human population size:

```sh
uv run --group fastdtwf python benchmarks/compare_startup.py \
  --output benchmarks/fastdtwf_pre_expansion_startup_cpu.json
```

This uses uniform input, three cold trials per configuration, and five warm
calls per trial. It also estimates totals for 1, 10, and 100 repeated calls.
The dependency group is not needed to install or use drifft. See
[comparison methodology and measured results](benchmarks/FASTDTWF.md).

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
