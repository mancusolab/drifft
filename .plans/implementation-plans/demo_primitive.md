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
