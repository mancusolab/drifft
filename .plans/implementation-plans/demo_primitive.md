# Demographic mathematical action primitives

Branch: demo_primitive. Authorized implementation of the agreed private HiJAX
history-action contract, with `_demographic_advance` and `_demographic_expect`
entry points and a shared oriented, augmented linear action.

- [x] Centralize power derivative identities; remove method-owned JVP hooks.
- [x] Implement backward epoch evaluation through transposed preparation factors.
- [x] Implement demographic action and linearized action with shared pairing pullback.
- [x] Connect public methods, preserve forward diagnostics, enable Krylov expectations.
- [x] Verify dense-reference values, JVP/VJP, sources, identity/boundary histories,
      batching, higher-order Sequential support, and numerical failure behavior.
- [x] Update documentation and review the final diff.

Geometry and numerical configuration are fixed; process parameters, initial
mass/terminal values, and singleton rate are differentiable. Numerical method
kernels approximate the common mathematical action; adaptive basis construction
is not the derivative target. No commit/push requested for this branch yet.

Verification completed:
- Demographic, Krylov sensitivity, and dense-backend regression run: 165 passed.
- Polynomial Krylov and epoch regression run: 76 passed.
- Additional identity-history parameter/vector derivative checks: 2 passed.
- Ruff lint/format, ty, strict clean documentation build, and git diff whitespace checks pass.
- Direct review covered the new primitive, transposed preparation factors, shared
  power rules, callers, numerical failure behavior, and documentation contracts.

Sequential reverse-over-reverse and forward-over-forward derivatives are checked.
Krylov higher-order parameter sensitivities still fail explicitly. Forward and
backward Krylov actions are independent approximations of the represented
operator, not exact transposes of the adaptive projections. User-owned demo.tsv
was left untouched. Changes remain uncommitted on demo_primitive.

Hygiene review:
- Clarified public differentiation targets, higher-order limits, diagnostic
  exclusion, and sensitivity failure behavior in demographic docstrings.
- Documented orientation and affine encode/decode transposition internally.
- Expanded the autodiff chapter with whole-history actions, the augmented
  source operator, both vector pullbacks, shared parameter pairing, and
  independent-reference validation guidance.
- Added offset-only/zero-rate pullbacks, tangent-only batching and pairing,
  successful-primal/failed-sensitivity behavior, and geometry rejection tests.
- Replaced the nested-JVP self-comparison with an independent dense oracle.
- All 33 demographic autodiff tests pass; lint, format, type checks, strict
  documentation build, and diff whitespace checks pass. No numerical behavior
  change was needed. Changes remain uncommitted.

Performance follow-up (in progress):
- First-trial benchmarks identified ~31% Sequential JVP and ~43% Jacobi/
  RationalKrylov gradient regressions relative to 18d4c3d.
- A wrapper-bypass experiment isolated RationalKrylov overhead to the history
  wrapper; the Sequential power JVP also discarded a reusable primal.
- Fuse the Sequential power primal/tangent recurrence and move checked history
  evaluation inside the primitive. Retain forward parameter-VJP residuals;
  preserve the opposite vector action and shared forward parameter pairing.
- Verify dense derivatives, higher/mixed derivatives, diagnostics, numerical
  failures, and batching, then rerun the focused benchmark. No new commit yet.

Performance follow-up completed:
- The action now owns checked primal evaluation. Explicit VJP forward/backward
  rules retain the original forward parameter pullback, replacing the separate
  linearized-history primitive that reconstructed its primal work.
- Sequential returns both outputs from its coupled recurrence while preserving
  a cached primitive boundary for higher derivatives.
- 244 relevant tests passed, including new mixed parameter/vector derivatives
  and diagnostic-only derivatives for an unconverged primal. Lint, formatting,
  type and whitespace checks pass. Direct review found no remaining correctness
  issues within this scope.
- All 18 benchmark outputs match the parent bit-for-bit; compiler temporary
  memory matches exactly. Fresh parent controls demonstrate timing drift;
  detailed results are in .cache/demo_primitive_benchmark/OPTIMIZATION.md.
- Optimization changes remain uncommitted; the earlier refactor is f9457fa.

Additional approved optimizations (items 1 and 4) completed:
- Joint expect parameter/vector pullbacks reuse the forward pairing value,
  retaining the opposite-action primitive boundary for higher derivatives.
- Backward epoch preparation returns its prepared problem as VJP auxiliary
  data, avoiding a second preparation while retaining process dependence.
- Removed the unused forward branch and renamed the backward history evaluator.
- 100 targeted tests passed, including a new dense-reference JVP of the joint
  expect gradient. Lint, formatting, type and whitespace checks pass.
- Focused joint-expect timing: Sequential 8.717 -> 7.929 ms; RationalKrylov
  31.254 -> 24.438 ms, with reduced compiler temporary memory. Single-trial
  details and raw outputs: .cache/demo_primitive_benchmark/EXPECT_REUSE.md.
- No changes to symbolic-zero handling or backward-linearization caching
  (the other discussed options). All follow-up optimizations remain uncommitted.

Symbolic-zero pullback optimization completed:
- Inspect the action cotangent before reconstructing inputs or materializing
  zeros. If all action cotangents are symbolic zeros, return symbolic input
  zeros without invoking parameter or vector pullbacks; diagnostics contribute
  no derivatives.
- A regression test failed before the change because a numerical pullback was
  invoked. Both orientations now pass, including nonzero diagnostic cotangents.
- Dense-reference stationary-point tests confirm numerical zero gradients retain
  nonzero curvature under JIT. No value-based zero detection was introduced.
- All 41 demographic autodiff tests, Ruff lint/format, ty, and whitespace checks
  pass. Direct review confirms the early return preserves the input PyTree and
  tangent avals. Changes remain uncommitted.

Symbolic-zero hygiene review:
- No correctness defects found in the early-return condition, diagnostic
  exclusion, or preservation of numerical-zero curvature.
- Construct symbolic input zeros directly from the primitive's declared avals;
  read the active-input mask only on the nonzero path.
- Extend the skipped-work test to cached primal inputs in both orientations,
  checking PyTree and tangent-type preservation for every input leaf.
- All 43 demographic autodiff tests pass, as do lint, formatting, type and
  whitespace checks. Changes remain uncommitted.

Backward parameter-linearization experiment (completed, uncommitted):
- Baseline revision: bb3bd60; source and measurement artifacts now live under
  `/tmp/drifft-expect-linearization`. A clean docs build cleared the initial
  `.cache` artifacts; the baseline source was recreated from git and its scalar
  measurements recovered from exact captured output. The report records this.
- Benchmark the public SD expectation likelihood gradient, ending 252 generations
  ago, with the documented Jacobi/RationalKrylov settings and fixed sampling
  reward. Record setup, compilation, synchronized execution, compiler storage,
  and peak RSS in fresh serial processes before and after the change.
- Retain each orientation's parameter linearization during its original primal
  evaluation. Keep opposite-action vector pullbacks and mathematical epoch rules.
- Check dense-reference derivatives, source offsets, batching, higher/mixed
  derivatives, symbolic zeros, and failure behavior. Compare small joint-gradient
  timings as well as the SD parameter-only workload before judging the tradeoff.
- The default SD baseline completed a first synchronized value-and-gradient:
  206.79 s, finite likelihood/gradient, 417.13 s compilation, 29.59 GiB compiler
  temporary storage, and 3.80 GiB process peak RSS. Hardware has 16 GiB RAM;
  compression/swap makes RSS understate the burden. Stopped untimed warmups
  rather than repeat the oversized allocation. No clean SD warm baseline exists.
- The orientation-local parameter VJP implementation passes all 46 demographic
  autodiff tests (including three new batched dense-reference cases), Ruff,
  formatting, and ty. Another 97 demography/power/symbolic-zero regression tests
  pass (143 total), along with the strict clean docs build and whitespace check.
- Changed SD: 203.64 s first call, 368.91 s compilation, 22.36 GiB compiler
  temporary storage, 6.77 GiB peak RSS. Maximum relative gradient difference is
  5.35e-9; log-likelihood differs by 3.87e-11. The allocation estimate falls 24.5%,
  but physical RSS rises and timing is memory-pressure-sensitive. No SD speedup
  or physical peak-memory improvement is established on this 16 GiB host.
- Small serial paired runs (30 synchronized samples) show parameter-only
  gradients improving 8.758 -> 6.688 ms for Sequential and 24.521 -> 16.188 ms
  for RationalKrylov, with lower temporary storage. Joint gradient timings are
  roughly unchanged; Sequential joint temporary storage increases 0.65%.
- Direct review found no additional correctness issue. Keep the simpler
  orientation-local linearization for its parameter-only benefit; investigate
  large SD storage separately. Full measurements, limitations, and reproduction
  scripts: `/tmp/drifft-expect-linearization/REPORT.md`.
- User-owned demo.tsv remains untouched. No commit made for this experiment.

Parameterized Jacobi operator experiment (in progress, 2026-11-05):
- User authorized implementation and testing without permission requests; no new
  commit or push requested. Baseline is bb87e5d, saved under
  `/tmp/drifft-parameter-operator/baseline`.
- Keep ordinary prepared Lineax operators as Equinox PyTrees; defer Jacobi
  assembly until the numerical solve. The private power primitive consumes an
  opaque HiType with a matrix cache and process/source-vector tangent space.
- Contract low-rank power adjoints directly into process parameters by streamed
  directional assembly; use an analytic contraction for encoded source columns.
  Retain mathematical rules, convergence checks, and higher-order limitations.
- Verify independent dense history derivatives, prepared custom solvers, shared
  operator accumulation, JIT, vmap in both AD orders, and compact cotangent shape.
- Compare current source, baseline, and assembly checkpointing in fresh serial
  workers. Use the previously selected smaller SD half and guard allocations.
  Benchmark files live in `/tmp`, outside the documentation build cache.

Parameterized operator implementation and measurements:
- Implemented a private lazy Lineax adapter, native operator HiType, parameter
  power/JVP/adjoint primitives, and a compact process/source cotangent space.
  Prepared custom solvers retain ordinary differentiable `mv`/`as_matrix` and
  Lineax materialization/solve behavior. Public API and mathematical AD targets
  remain unchanged.
- Dense-reference and direct-power tests cover shared operator accumulation,
  sources, both orientations, parameter-only and joint gradients, tangent-only
  batching, mapped gradients, nested parameter grids, and custom Lineax solves.
  Added native-path coverage for higher-order rejection and sensitivity failures.
- Nested mapping initially failed because HiJAX shifts mapping specs with axis
  arithmetic. The mapping spec now records/shifts its payload axis; nested
  primal and gradient tests pass. No dependency/environment changes were made.
- All 52 initial focused tests passed; 15 follow-up edge cases passed; the added
  nested-gradient check passed. Ruff, formatting, ty, and strict docs build pass.
  The complete 910-test suite is running; final outcome follows below.
- Smaller SD half (26 intervals, 5672 -> 252 generations ago), fresh serial CPU
  workers, dimension256, rtol1e-9/atol1e-11, two warm samples:
  baseline 34.786 s / 5.973 GiB compiler temporaries / 5.521 GiB peak RSS;
  HiType 44.067 s / 2.799 GiB / 5.857 GiB;
  coarse assembly checkpointing 36.740 s / 7.298 GiB / 5.695 GiB.
  Maximum relative HiType gradient difference is 5.28e-11. This establishes a
  temporary-storage/runtime tradeoff, not a physical peak-memory improvement.
- Small source-inclusive parameter gradients: 16.128 -> 25.816 ms, temporaries
  5,744,032 -> 2,542,848 bytes; joint gradients: 27.815 -> 33.803 ms,
  temporaries 8,036,192 -> 4,984,512 bytes. Four-direction forward mode:
  46.372 -> 49.091 ms, temporaries 8,720,848 -> 8,561,288 bytes.
- Raw timings, source controls, scripts, outputs, and methodology are preserved
  in `/tmp/drifft-parameter-operator/REPORT.md`. Recomputing process directions
  is the present speed cost; HiType wrapping alone is not the memory optimization.
  Changes remain uncommitted for review. User-owned demo.tsv is untouched.

Final verification for the parameterized operator experiment:
- `pytest -p no:capture -q`: 910 passed in 579.40 s.
- Ruff checks and formatting, ty, strict Zensical build, and diff whitespace
  checks pass. Direct review covered logical versus numerical payloads,
  source/transposed contractions, parameter reconstruction, geometry exclusion,
  custom solver compatibility, symbolic zeros and batching. No remaining
  correctness finding was identified within the tested scope.
- The candidate is a storage/runtime tradeoff: approximately 53% lower compiler
  temporary storage and 27% slower warm SD gradients, with slightly higher
  measured process peak RSS. Keep this distinction when deciding whether to
  adopt the experiment; the current directional contraction is not a speed win.
- Preserved the final implementation source under
  `/tmp/drifft-parameter-operator/candidate/src` alongside baseline/checkpoint
  controls. No commit, push, environment change, or permission request was made.

Adoption decision (2026-11-06):
- User accepted the measured storage/runtime tradeoff and requested committing
  the implementation on demo_primitive. Preserve the benchmark limitations above.
