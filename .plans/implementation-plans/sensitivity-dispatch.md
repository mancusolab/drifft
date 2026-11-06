# Separate numerical sensitivity selection from power autodiff

Branch: `refactor/sensitivity-dispatch`

## Scope and contracts

- Keep history and power derivatives defined by the mathematical operations.
- Let built-in method capabilities select sensitivity evaluation and enforce
  numerical acceptance; keep kernels in `sensitivity.py` and primitives in
  `autodiff.py`.
- Keep operator pushforwards and parameter contractions in their adapters.
- Preserve Sequential's fused recurrence and higher derivatives, Krylov cached
  values/matrices, symbolic-zero shortcuts, batching, and convergence failures.
- No new requirements on public custom epoch methods; no new sensitivity-object
  hierarchy, numerical algorithm changes, or unrelated transformation fixes.

## Tasks

- [x] Create branch and inspect existing dispatch and test coverage.
- [x] Replace solver-family checks with private method capabilities.
- [x] Move coupled recurrence to sensitivity kernels and numerical acceptance
  into Krylov method capabilities; update module-level explanations.
- [x] Add independent-evaluator coverage for dispatch and state-only shortcuts.
- [x] Run power, parameterized power, sensitivity, and history AD tests.
- [x] Compare baseline/candidate compiled costs and memory for forward/reverse AD.
- [x] Review diff and run lint, formatting, type, and documentation checks.

Verification artifacts: `/tmp/drifft-sensitivity-tests.log`,
`/tmp/drifft-sensitivity-before.json`, `/tmp/drifft-sensitivity-after.json`.

## Verification evidence

- Independent-evaluator regression: 2 cases passed (with and without a supplied
  primal), including state-only derivative bypass.
- Ruff lint/format, ty, diff whitespace checks, and strict Zensical build pass.
- Direct diff review found no additional actionable issues. Removed the numerical
  configuration protocol entirely in the follow-up;
  sensitivity kernels now accept explicit keyword settings.
- Six baseline/candidate probes (Sequential, polynomial Krylov, rational Krylov;
  forward and reverse AD) have identical values/gradients, compiler cost-analysis
  dictionaries, and compiler temporary-memory sizes. These are small compiled
  workload comparisons, not full demographic timing benchmarks.
- Existing focused suite: 149 passed in 235.01 seconds across power actions,
  parameterized powers, Krylov sensitivities, and demographic autodiff.
- Final power-action suite including new cases: 23 passed in 23.54 seconds.
  Together these runs cover 151 distinct passing tests.

## Follow-up: explicit numerical settings

- Removed `_KrylovConfig` and its unused typing imports.
- Sensitivity kernels accept keyword-only dimension, tolerances, and polynomial
  mode. Projection helpers accept only dimension and mode.
- Updated built-in method calls and direct numerical-kernel tests.
- Repeated six compiled probes: all values, gradients, compiler costs, and memory
  fields exactly match the pre-refactor baseline.
- Focused tests: 100 passed in 133.46 seconds (Krylov sensitivities, power
  actions, and parameterized powers); log: `/tmp/drifft-explicit-sensitivity-tests.log`.
- Ruff, formatting, ty, whitespace checks, and strict documentation build pass.

## Hygiene review

- Reviewed the complete diff, call sites, stale-name references, convergence
  checks, primal reuse, symbolic-zero handling, and derivative test coverage.
- No correctness findings. Clarified kernel settings and linearization return
  ordering; added reverse-mode coverage to the independent-evaluator regression.
- Both independent-evaluator cases pass with JIT JVP/VJP comparisons and the
  state-only shortcut check. Repository src/tests lint and formatting, ty, strict
  docs build, and whitespace checks pass. Numerical implementation unchanged
  during this pass; the full repository test suite was not rerun.
