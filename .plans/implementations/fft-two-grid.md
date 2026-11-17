# Production two-grid FFT integration

Authorized by the user's request to refactor FFT to the researched architecture.
Work on the current `fft_diagnostic` branch; preserve unrelated documentation and
`demo.tsv`. No new public configuration or diagnostic classes.

## Contract

- Keep `atol`, `grad_atol`, `angular_sensitivity`, cached reports, linearity,
  transpose pairing, and branchwise derivative scope.
- Start convolution at `h = 1/(4 V)` and direct cutoff at 32 copies. Select
  dyadic interpolation spacing `H = k h` with the existing analytic bounds.
- Build centered, cancellation-resistant factors in `src/drifft/_backend/fft.py`.
  Check input curvature on the final tail window; if needed refine convolution
  while preserving H. Retain only coarse output nodes and gather strided results.
- Preserve explicit interpolation-order override semantics. Automatic planning
  retains finite order/cutoff fallback. Arithmetic error remains unbounded in the
  shared report; integration does not implement the note's future certified
  floating-point/cost-optimization search.
- Keep the research baseline reproducible without a runtime dependency on it.

## Tasks and evidence

- [x] Regression tests for production factor identities, stable cumulants,
  strided padding/gathers, conditioning and requested tolerance actions.
- [x] Integrate host grid planning, factors, coarse metadata and bucket gathers.
- [x] Adapt research construction to the new private layout; compare production
  against independent Decimal rows and the historical architecture.
- [x] Update FFT API/method docs and research status.
- [x] Run numerical/AD/backend integration tests, lint, format, type checks,
  strict docs build, and review the complete diff.

## Current evidence

- Production at M=8192/32768 passed 61-query Decimal row and derivative
  comparisons, adversarial signed actions, and transpose actions (2 tests).
  Worst angular-gradient row errors: 8.69e-12 and 2.66e-11 at grad_atol=1e-10.
- Focused non-slow tests: 165 passed; the remaining derivative test's tolerance
  fixture now requests gradient planning to resolve direct tails. The corrected
  test passed in the full shard, with midpoint second-derivative error 1.18e-9.
- CI shard 1: 300 passed, 837 deselected (292.21 s). The two warnings came from
  existing invalid-static-JAX-array tests. Shard 2: 837 passed, 300 deselected
  (686.01 s). Together these cover all 1,137 collected tests, including the large
  production accuracy checks; no skips or failures.
- The original baseline at M=512/1000 remains bitwise identical to arrays built
  by a1c2af9; checked by loading that source independently from git.
- Repository Ruff, formatting, ty and strict Zensical build passed after the
  production, documentation and fixture edits.
- Wide retained windows exposed overflow in the old log-range balancing. The
  production builder now uses max(b)=1, verified with full-support factors.
- Direct review covered the full source diff, helper callers, strided padding,
  final-window refinement, budget/report consistency, and documentation. No
  unresolved in-scope code issues remain. All planned work is complete.

Commands from `/Users/nicholas/Projects/drifft` always run tests with
`.venv/bin/python -m pytest -p no:capture`. Target FFT bounds, derivatives,
spacing study and backend accuracy first; broaden to transition/evolution and
CI shards as justified. Use `.venv/bin/ruff`, `.venv/bin/ty`, and
`.venv/bin/zensical build --strict` for repository checks.

## Follow-up hygiene pass

- Reproduced inaccurate/NaN centered cumulants for large shifts against Decimal
  references. Production now uses logaddexp outside the small-shift series range;
  checks span shifts from -1000 to 1000 and still resolve quadratic terms near zero.
- Replaced the temporary Equinox block with a NumPy NamedTuple and pack spectra
  on the host before conversion, eliminating per-block JAX/NumPy round trips.
  Fine-grid spacing is derived from the prepared interpolation spacing and stride.
- Documented all mathematical helpers, clarified geometry units and host/runtime
  boundaries, removed repeated reductions in accuracy tests, and updated the
  numerical guide, factor notation and historical-note links.
- Verification after cleanup: the 292-test FFT/backend/transition selection passed
  in 141.22 s. Six final targeted checks passed, including host factorization with
  JAX x64 disabled and both production/historical nonfinite-factor rejection.
  Large-case errors remain 8.69e-12 and 2.66e-11 for angular-gradient rows.
- Repository lint, formatting, type checks, strict documentation build and
  whitespace checks passed. AST inspection found no undocumented functions or
  classes in the five reviewed Python files. No new module or public API was
  introduced; unrelated user changes were preserved. No unresolved findings.

## Retire the trial implementation

Authorized after production integration and the hygiene pass. Keep the notes and
recorded results; remove the prototype builders, monkeypatching, sweeps and CLI.
Move the Decimal oracle into `tests/_reference.py` and retain production numerical
regressions as `tests/test_fft_numerics.py`. Preserve the original large-case
physical probes so the measured regression errors remain comparable. Point
ongoing performance comparisons at `benchmarks/compare_backends.py`; document
the historical commit needed to reproduce the original experiment.

- [x] Move the reference and adapt retained tests to production-only construction.
- [x] Remove the trial script and its CLI/prototype-only tests.
- [x] Update documentation and verify retained numerical checks, reference users,
  formatting, lint, types and the strict docs build.

Retirement verification: 193 selected tests passed in 148.76 s, covering FFT
numerics/bounds/derivatives, shared backend accuracy and epoch integration.
Two direct-only cases were rechecked after an explicit JAX input-conversion fix
for the type checker. CI shard 2 collects all 26 retained numerical regressions.
The moved Decimal reference and both 61-query sets match their pre-removal
snapshots exactly. Hashes confirm that production FFT source and the archived
results were untouched by retirement. Lint, formatting, ty, strict docs and
whitespace checks passed. Active code has no dependency on the retired driver;
its only remaining references are commands explicitly scoped to commit c4e07c7.
