# Modular demographic epoch methods

Status: implementation complete. User approved the design in this conversation.

## Scope

Separate Jacobi geometry from the rational Krylov algorithm. Keep Equinox
abstract/final modules and composition. DemographicTransition is the public
composition point; do not add an Epoch class, a second public evolve entry point,
an observation likelihood framework, an exposed linear-solver option, or HSS.
Implement the existing Sequential and RationalKrylov algorithms; polynomial
Krylov remains future work, not an advertised placeholder.

## Acceptance criteria

- AC1: Jacobi owns size, rank/padding, selection domain, and geometry.
  RationalKrylov owns dimension and power tolerances, without population geometry.
  Sequential is a geometry-free method. An abstract Equinox method interface and
  internal prepared problem separate representation from power computation.
- AC2: DemographicTransition accepts backend, optional boundary_backend, method
  (default Sequential), and frequency_map. FFT/NUFFT default to their own boundary
  factory. Jacobi requires an explicit FFT/NUFFT boundary factory. Validate
  unsupported combinations at construction, including histories with no holds.
- AC3: Supported initial combinations: FFT/NUFFT + Sequential, Jacobi +
  RationalKrylov. Jacobi + Sequential may be provided if it shares the prepared
  problem and preserves the same reduced-model semantics. Do not silently
  materialize FFT/NUFFT matrices or implement iterative rational solves.
- AC4: Preserve forward and segregating chronology, singleton immigration after
  every generation, identity behavior, input validation, float64, JIT, batching,
  and differentiability of selection and mass/source. Preserve matched expect
  for FFT/NUFFT Sequential; reject unsupported adjoints explicitly.
- AC5: Use Lineax internally for the repeated shifted solves, with one reusable
  factorization per power action. Preserve the existing power diagnostics and
  make inner-solve failure observable. Do not stop gradients through the outer
  Krylov basis. Do not attach factorization/basis state to returned results.
- AC6: Demographic forward methods return an explicit result containing
  distribution, converged, status, and per-interval diagnostics; throw=False
  permits inspection. No invented total-demography error certificate.
- AC7: Migrate existing tests, maintained benchmark scripts, and docs to the new
  API; document the breaking migration of experimental epoch constructors,
  epoch_backend, and demographic array returns. Keep Transition unchanged.
- AC8: Leave pre-existing CI edits and research documents/data intact. Changes
  to research scripts are limited to API migration needed for reproduction.

## Execution tasks

1. Add failing independent-reference tests for the new composition API, including
   selected changing-size histories, immigration, failure status, derivatives,
   batching, rejected combinations, and identity histories.
2. Extract Jacobi and implement method/prepared-problem contracts, Lineax shifted
   solves, and demographic orchestration. Migrate existing source tests.
3. Migrate benchmark callers and numerical/user API documentation.
4. Run tests with pytest -p no:capture, Ruff, type checks, documentation build,
   and package build. Review changes and fix findings before completion.

## Review profile

Numerics and public API. All numerical source stays Functional Core; benchmark
drivers stay Imperative Shell. No new inference model is introduced.

## Progress

- [x] Numerical implementation and source tests
- [x] Benchmark and documentation migration
- [x] Verification and independent review

## Verification

- Full suite: `pytest -p no:capture -q`: 241 passed.
- Memory-study checks: six HSS tests passed; the retention test passed after
  rerunning with permission to invoke `ps` outside the sandbox.
- Ruff lint and formatting, type checking, and `git diff --check` passed.
- Strict documentation build and all seven Python examples in the demographic
  and constant-size method guides passed.
- Source and wheel builds passed; public imports from the built wheel passed.
- Baseline and numerical/API specialist reviews found no actionable issues.

Performance benchmarks were not regenerated. Historical reports are marked as
predating this refactor. RationalKrylov remains experimental; unsupported method
and backend combinations are rejected explicitly.
