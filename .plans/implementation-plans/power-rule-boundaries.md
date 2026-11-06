# Power-rule boundaries

Status: complete; implemented, reviewed and verified; commit authorized.
Implementation authorized; two context-free plan reviews completed before source edits.
Base: 7bf9e48 on demo_primitive. Preserve untracked user-owned demo.tsv.

## Objective and invariants

Keep the represented mathematical history action as the derivative target for
both advance and expect, including parameter-dependent encoding, reconstruction,
boundaries and sources. Separate A(theta)^k x differentiation from the operator's
parameter-to-action derivative and from numerical power evaluation. No public API
changes. No AD through adaptive Krylov bases, matrix caches, or fixed geometry.

Preserve Sequential higher derivatives, Krylov first-order limits, source offsets
at zero rates, signed vectors, shared parameter aliases, symbolic-zero activity,
compact parameter cotangents, one-at-a-time active direction contraction, minimal
geometry, shared batch geometry, and cached primal reuse. Do not force full-state
or matrix-free operators into a dense matrix or small parameter vector.

## Proposed internal organization

- New _evolution/operators.py owns generic numerical-operator access,
  operator pushforward and fixed-factor contraction. Use a numerical_operator
  singledispatch on PREPARED Lineax operators, defaulting to identity. Jacobi's
  registration lives beside _JacobiOperator and materializes its reduced matrix.
- Cached-power binding is a second singledispatch in autodiff.py, also on PREPARED
  Lineax operators, with the Jacobi registration colocated there. It binds either
  ordinary PyTree coordinates or the opaque parameter representation. Compact
  primitives use explicit compact pushforward/contraction helpers after binding.
  There is no strategy object mixing prepared and bound representations and no
  representation-to-autodiff callback/import cycle.
- Generic pushforward uses TangentLinearOperator; generic contraction uses a
  filtered operator-action VJP. Compact pushforward assembles a reduced direction;
  compact contraction streams active process directions and contracts source
  coordinates analytically. Both consume fixed adjoint factors; neither owns
  history or power differentiation rules.
- parameterized.py keeps _JacobiOperator, _Operator/_OperatorType, _Spec, pack/
  unpack/transpose primitives, activity extraction and Jacobi derivative strategy.
  Matrix assembly remains in _backend/jacobi.py. No power or sensitivity numerical
  kernels remain in this module.
- autodiff.py owns both sets of power/sensitivity bindings. Retain distinct ordinary
  PyTree and compact HiType bindings; do not force them through eqx.partition as
  though their primal/tangent structures matched. Move _Power, _Sensitivity,
  _Pullback and wrappers here; make _Sensitivity a HiPrim sibling rather than
  inheriting incompatible power behavior. Use shared functions where genuine
  contracts match, not inheritance merely to reuse helpers.
- sensitivity.py retains numerical Frechet actions and adjoint factors. Separate
  factor computation from generic operator contraction (currently combined in
  _frechet_operator_pullback). Share convergence checking/action evaluation between
  bindings without exposing factor construction to outer differentiation.
- methods.py obtains numerical representation and cached binding from the selected
  capability, removing concrete _JacobiOperator checks. preparation.py still owns
  representation selection and returns ordinary differentiable Lineax operators,
  preserving custom methods and direct preparation differentiation.
- Import direction: operators.py depends on Equinox/Lineax/JAX;
  parameterized.py depends on operators.py; sensitivity.py depends on numerical
  projection/problem code; autodiff.py depends on those three; methods.py imports
  binding and numerical access. Jacobi numeric registration is loaded by the
  explicit autodiff/preparation imports of parameterized.py.

## Execution phases

1. Freeze baseline and review plan.
   Snapshot src to /tmp/drifft-power-boundaries/baseline/src. Save small benchmark
   scripts/results and current half-SD record. Two independent context-free agents
   inspect this plan and repository for mathematical/AD risks and organization/
   performance risks. Reconcile findings in this file before source edits.
2. Establish operator differentiation contract.
   Extract generic factor contraction and Jacobi pushforward/contraction, retaining
   exact arithmetic and map order. Add adapter-level checks against dense analytic
   scalar-pairing derivatives, including matrix-free fallback, transpose, source,
   partial activity and shared parameter aliases. Retain direct preparation tests.
3. Reorganize bindings and solver selection.
   Move power classes to autodiff.py, update internal imports/tests, route numerical
   operator/cached binding through capability selection, remove concrete backend
   checks from methods.py. Keep Sequential's coupled recurrence separate. Delete
   superseded helpers rather than leave compatibility aliases for private names.
4. Verify semantics, storage, performance; review.
   Run targeted tests first, then complete suite with pytest -p no:capture. Run
   Ruff lint/format, ty and git diff --check. Review diff for accidental scope drift,
   new retained arrays, dense fallback, lost activity, imports and error behavior.
   Compare same small benchmark before/after in serial fresh processes, and one
   selection-only smaller-half SD execution per variant using existing script.
   If a material slowdown/storage increase appears, investigate before adopting;
   single-run timing noise is not a regression verdict. Update internal docs only
   to describe final boundaries; no migration language. No commit/push requested.

## Acceptance and evidence

- Both history orientations and generic/compact power derivatives match independent
  dense references: test_demographic_autodiff.py, test_parameterized_power.py,
  test_power_action.py, test_krylov_derivatives.py, test_krylov_frechet.py.
- Sequential higher derivatives and Krylov failure contracts remain covered.
- Adapter unit tests establish parameter pushforward and fixed-factor contraction;
  generic test uses a matrix-free operator whose as_matrix fails, ensuring no
  accidental materialization. No zero-sized or source-only regression.
- Existing instrumentation verifies only active process coordinates assemble.
- HiType cotangent shape, geometry payload and nested batch sharing checks persist.
- Preserve exact source adjoint formulas in both orientations and avoid treating
  factors as parameter-dependent inside contraction.
- Small storage probe and compiler memory analysis show no reintroduced matrix
  cotangent/geometry replication. Compare numerical output and gradient alongside
  timings; compilation and execution are separate metrics.
- Full suite, lint/type checks and focused direct review required before completion.

## Out of scope / known limitations

Do not change demographic history rules, add new public types, add higher-order
Krylov support, reuse left projections yet, or fix staged batched-VJP / reverse AD
through batched scan limitations. Those failures predate this work; record any
change to their behavior rather than silently broadening scope.

## Progress / review decisions

- Plan drafted; no source changes yet.

Plan-review resolution (before source edits):
- Organization reviewer identified an avoidable prepared/bound strategy ambiguity
  and callback dependency cycle. Adopt two narrow prepared-operator dispatchers
  and explicit bound-coordinate helpers instead of a new strategy hierarchy.
- Mathematical reviewer requires alias/activity ownership and orientation to be
  explicit: when visible parameter leaves are provided, they alone supply the
  tangent and active mask (do not count the opaque tangent again); the opaque
  primal remains live for mixed derivatives. Transposition changes matrix/spec
  orientation, not parameter coordinates. Keep both invariants and tests.
- Both context-free reviews complete; no mathematical blocker. Preserve supported
  mixed state/parameter derivatives, not a blanket rejection of second derivatives.
  Add compact cached-value tangent independence coverage. Preserve pre-factor
  inactive short circuit and use the matrix already assembled for projection.
- Convergence checks stay in shared power-binding helpers, while numerical kernels
  return estimates/status and factor contractions receive fixed factors only.
- Phase 1 complete. Baseline snapshot and selection-only small benchmark saved
  under /tmp/drifft-power-boundaries. Implementation begins after review resolution.

Implementation progress:
- Phase 2 implemented: generic numerical/pushforward/factor-contraction helpers in
  operators.py; compact assembly pushforward, streaming contraction and activity
  extraction in parameterized.py. Six new adapter/cached-binding tests pass,
  including forbidden matrix materialization and repeated parameter aliases.
- Phase 3 implemented: all power/sensitivity bindings now in autodiff.py, with
  shared checked Frechet/factor kernels; compact sensitivity is a HiPrim sibling.
  Cached binding dispatches on prepared operator type, numerical projection uses
  the separate representation dispatcher. methods.py has no Jacobi-specific check.
  Removed old combined numerical-factor/operator-contraction helper; its tests now
  exercise factor computation and contraction separately.
- Internal module documentation explains ownership. No public API or history-rule
  changes. Full verification and performance comparison remain pending.

- Targeted regression: 98 tests passed in 131.90 s; all six new boundary tests
  passed in 19.42 s. Full suite is running against the final functional changes.
- Direct review checked generic matrix-free identity adaptation, one-time numerical
  Jacobi assembly, inactive pre-factor guard, cached tangent ownership, transpose
  parameter coordinates, source contraction orientation, generic None filtering,
  sibling sensitivity transformation hooks and import direction. Removed an unused
  protocol left behind by the cached-binding signature change. Ruff/type checks pass.
- Complete test suite: 941 passed in 621.92 s. Ruff lint/format, ty and whitespace
  checks pass. Serial performance/storage probes started after suite completion,
  with fresh processes, one BLAS thread and no compilation cache.

Final verification and adoption evidence:
- Serial small benchmarks: selection-only 19.938 -> 17.185 ms; joint parameter/state
  30.711 -> 30.904 ms. All outputs/gradients and compiler storage match exactly.
- Retained storage unchanged: 4,776-byte unbatched pullback, 256-byte shared batch
  geometry. This confirms preservation, not a new memory optimization.
- Fresh half-SD pair, one selection-only execution per variant: 33.813 -> 33.058 s;
  compilation 61.303 -> 59.756 s; compiler temporaries identical at 3.151 GiB.
  Values and gradients identical. No material regression observed; no speedup
  claim from a single pair. Detailed report: /tmp/drifft-power-boundaries/REPORT.md.
- All planned phases complete. Source changes are limited to power/operator
  boundaries and internal documentation. No history/public API/solver algorithm
  changes. User-owned demo.tsv untouched. No commit or push requested/performed.

Final sanity pass and commit authorization:
- User requested a final sanity pass and commit. Direct review found no additional
  correctness issue. Clarified numerical-cache ownership in the cached-binding
  docstrings and wrapped the module overview; no numerical changes after the
  941-test run and benchmark comparison. Final lint/type/format checks repeated.
- Commit includes the implementation, rationale comment, tests and this plan.
  User-owned demo.tsv remains excluded; no push requested.
