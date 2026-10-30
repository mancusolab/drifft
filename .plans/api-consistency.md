# Public API consistency

## Context and authorization

- Repository: `/Users/nicholas/Projects/drifft`.
- Starting branch/commit: `main` at `c82f213`.
- User requested a durable plan for the API review findings, with review item 1
  (the public epoch-method extension boundary) addressed last.
- Implementation authorized by the user's subsequent "ok lets begin".
  User subsequently authorized committing the completed refactor and hygiene
  changes locally on main. No push or branch creation is requested.
- User selected `segregating=True` on existing methods and a minimal supported
  public epoch-method extension interface. Keep that extension milestone last.
- User clarified during the hygiene pass: no API migration guide or backward
  compatibility is needed. Document the current API directly and remove the
  migration page. This supersedes earlier migration-documentation instructions.
- All six milestones are implemented and verified. Accepted choices and actual
  verification are recorded below; no design decision remains pending.
- Earlier architecture plans are historical. This plan owns the API changes
  listed below and intentionally permits breaking changes without compatibility
  aliases, following the user's established policy for this new repository.

## Goal and constraints

Make the public API easier to compose and predict, without changing numerical
algorithms or adding runtime overhead merely to normalize interfaces.

- Retain the separation between frequency maps, backend geometry, transition
  operations, and evolution algorithms.
- Preserve float64, dynamic biological parameter leaves, JIT/batching contracts,
  adjoint relationships, supported derivatives, and explicit unsupported-AD errors.
- Preserve full/interior state layouts, signed mass, rectangular transitions,
  source timing, zero-generation behavior, and demographic chronology.
- Keep arbitrary-probability transitions and independent Dense references.
- Preserve the accepted Lineax banded solver and factor reuse.
- Change docs, maintained benchmark callers, and tests alongside each API change.
  Preserve the biological meaning of examples when parameter units change.
- No new source module or package is currently justified. Keep result/status
  types near their computations and shared helpers with their owning modules.
- Preserve unrelated untracked files: `benchmarks/POWER_LOOPS.md`,
  `benchmarks/compare_power_loops.py`, `benchmarks/power_loops_cpu.json`,
  `benchmarks/power_unroll_cpu.json`, and `response.md`. If a necessary caller
  migration reaches one, explain and isolate that migration; do not stage the
  research artifact as part of this work.

## Ordered milestones

### 1. Align solve failure conventions — verified (review item 2)

Dependencies: none.

Inspect/change `_operator.py`, `_solve.py`, `__init__.py`, occupation callers,
`docs/guide/solves.md`, API reference, and related examples/benchmarks.

Implemented contract:

- Add `throw=True` to `Transition.expected_visits()`, matching evolution's
  exception-first default. `throw=False` retains a candidate and diagnostics.
- Make the domain `status` describe the complete occupation calculation,
  including the recomputed residual; successful status and `converged` agree.
- Keep raw Lineax information explicitly as `solver_status` and `stats`.
  A small occupation-specific enum in `_solve.py` is preferable to forcing
  occupation outcomes into `EvolutionStatus`; do not add a new status module.
- Static configuration errors continue to raise. Document whole-batch throwing
  and distinguish solver failure, nonfinite output, and unacceptable residual.
- Retain the existing solver choices/tolerances for this milestone. Replacing
  string choices with a public solver-object interface is outside this plan.

Acceptance/verification:

- Both solvers match independent dense occupation values and residuals.
- A low iteration limit returns failure with `throw=False` and raises with
  `throw=True`; existing tests requesting failed candidates pass `throw=False`.
- A solver-success candidate rejected by the independent residual check has an
  unsuccessful domain status. Exercise this boundary with a controlled solver
  result if real solver termination cannot reproduce it deterministically.
- Check finite/nonfinite failure behavior under JIT and batched nonthrowing use.
- Run `tests/test_solve.py`, `test_transition.py`, `test_validation.py`, affected
  occupation cases in `test_dense_backend.py`, `test_spectral.py`,
  `test_jacobi_backend.py`, and `test_demography_benchmark.py`.

### 2. Align directional selection coefficient units — verified (review item 3)

Dependencies: milestone 1 verified; no numerical dependency on its implementation.

Inspect/change `_frequency.py`, Jacobi's `_selection_parameters` in
`_backend/jacobi.py`, selection tests, guide/model/math pages, examples, and
benchmark entrypoints that construct or update selection maps.

Implemented contract:

- Define `AdditiveSelection(s)` using the derived-homozygote fitness reduction,
  giving genotype fitnesses `(1, 1-s/2, 1-s)`.
- Then `AdditiveSelection(s)` agrees with `DominanceSelection(s=s, h=0.5)`;
  recessive and dominant selection already use the homozygote reduction `s`.
- Keep independent penalties `s_het`, `s_hom` on `DiploidSelection` and retain
  biologically distinct strength/multiplier parameters on the other models.
- Rename the additive parameter leaf to `s`, including `eqx.tree_at` callers.
- For each existing example intended to keep its original fitnesses, replace
  old `s_het=a` with new `s=2*a`. Inspect positional uses as well as keywords.
  Independently update derivative expectations for the changed parameter units.

Acceptance/verification:

- Compare named models against independent genotype-fitness formulas, including
  negative coefficients and endpoint behavior.
- Verify the additive/dominance equivalence in values and derivatives with the
  same new coefficient, and old/new biological equivalence with `s=2*s_het`.
- Jacobi derives `(s/2, s)`, applies the correct bounds, and matches independent
  Dense calculations for composed mutation/selection evolution.
- Run `test_frequency_maps.py`, `test_jacobi_diploid.py`,
  `test_jacobi_backend.py`, `test_dense_backend.py`, and relevant gradient cases.
- Search maintained source/docs/callers for obsolete additive `s_het` uses;
  preserve legitimate `DiploidSelection.s_het` references.

### 3. Clarify factories and transition construction — verified (review item 4 and minor redundancy)

Dependencies: milestones 1–2 verified.

Inspect/change `_demography.py`, `_operator.py`, maintained constructor callers,
API docs, guides, and benchmark integration tests.

Implemented contract:

- Rename demographic constructor arguments to `backend_factory` and
  `boundary_backend_factory`; `Transition.backend` remains a prepared instance.
- Retain factory inputs in gene-copy units and history sizes in diploid Ne.
  Make these units explicit in signatures/docstrings and examples; do not
  change the schedule representation in this cleanup.
- Retain `Transition(backend, probabilities=...)` as the canonical explicit-row
  constructor and remove the forwarding `from_probabilities()` classmethod.
- Keep map/probability mutual exclusion, rectangular sampled rows, and current
  complete-grid preconditions for segregating/occupation operations.
- Defer whether epoch methods consume a `Transition` instead of backend/map
  arguments to milestone 6: that decision belongs to the extension boundary.

Acceptance/verification:

- Factories still prepare one geometry per distinct size, and a shared hold/
  boundary factory still reuses geometry.
- All old maintained keyword/classmethod calls are migrated with no aliases.
- Independent references continue to verify rectangular operations, transpose
  identities, source-size inference, explicit-row differentiation, and histories.
- Run `test_transition.py`, `test_demography.py`, `test_modular_methods.py`,
  `test_dense_backend.py`, plus affected encoding/epoch/demography benchmarks.

### 4. Normalize segregating evolution and source naming — verified (review item 5)

Dependencies: milestones 1–3 verified.

Inspect/change `_operator.py`, `_demography.py`, `_evolution/methods.py`, callers,
and the transition/epoch/demography guides and API references.

Implemented contract:

- Use the user-selected `segregating=False`
  on `advance` and on currently supported transition `expect` operations,
  matching the existing epoch interface and avoiding extra method entrypoints.
- Rename demographic singleton input to `singleton_mutation_rate`. Keep epoch
  `immigration` for arbitrary source vectors; their units and responsibilities
  differ. Reject an explicitly supplied singleton rate in full-state mode.
- Keep `Mutation.forward_rate`/`backward_rate` for recurrent mutation of existing
  alleles. Document the source/transition distinction with one concise example.
- Normalize existing capabilities only. Do not add Krylov expectation
  transposes, demographic interior expectations, or generalized source schedules.

Acceptance/verification:

- Independent recurrences verify endpoint discarding and source addition after
  every generation, including size-changing boundaries and zero generations.
- Full/interior shapes, singleton rate gradients, signed immigration, JIT,
  batching, and supported higher derivatives remain correct.
- Run `test_transition.py`, `test_epochs.py`, `test_epoch_validation.py`,
  `test_demography.py`, `test_modular_methods.py`, `test_dense_backend.py`, and
  relevant Krylov demographic gradient cases.
- Search all maintained callers/docs for removed suffixes/keywords after the
  chosen convention is applied.

### 5. Simplify precision arguments and improve signatures — verified (review items 6–7)

Dependencies: milestones 1–4 verified so annotations describe settled signatures.

Inspect/change `_backend/fft.py`, `_backend/nufft.py`, `_frequency.py`,
`_evolution/methods.py`, public API tests, and signature-rendering templates as
needed for the actual annotation changes.

Implemented contract:

- Remove `dtype` from FFT/NUFFT constructors. All backends retain the existing
  float64 policy and reject disabled x64 at preparation time.
- Annotate numerical parameter converters with accepted scalar input types and
  array return types so generated constructors expose useful input annotations
  instead of `Any`. Keep their stored fields as JAX arrays.
- Use `AbstractTransitionBackend` rather than `eqx.Module` for backend arguments
  on epoch interfaces where that is the actual accepted contract.
- Preserve the existing rendering of public class names in docs. Avoid changing
  class `__module__` values or introducing runtime aliases to prettify signatures.
- Keep distinct accuracy controls (`tail_tol`, `accuracy`,
  `one_step_tolerance`, method `rtol`/`atol`); they control different errors.

Acceptance/verification:

- Inspect representative generated constructor/method signatures and rendered
  docs. Python scalar and JAX scalar inputs remain accepted, with dynamic leaves.
- Invalid vector parameters still fail; dtype and x64 behavior remain consistent.
- Run `test_public_api.py`, `test_validation.py`, `test_frequency_maps.py`, and
  affected backend construction cases. Run type checks and a docs build.
- Do not add tests that simply duplicate documentation wording or every rename.

### 6. Resolve the public method extension boundary LAST — verified (review item 1)

Dependencies: all five earlier milestones verified. This milestone must not be
pulled forward during earlier renames or preparation cleanup.

Inspect `_evolution/{methods,problem,preparation,autodiff}.py`, `_demography.py`,
`__init__.py`, API reference, and the custom-method example in
`tests/test_modular_methods.py`.

Implemented contract:

- Support custom methods through a public read-only `EpochProblem` protocol and
  `prepare_epoch()`. Keep the existing backend/map epoch calling convention.
- The protocol describes the working operator, initial state, exponent,
  reconstruction, error norm and shared result checks. It introduces no runtime
  wrapper or public construction of projection bookkeeping.
- Keep representation requirements explicit, and make direct epoch calls and
  demographic holds use the same extension contract.
- Keep matched expectations explicitly Sequential-only. Preserve current Krylov limitations;
  never infer linearity from the method name or a generic autodiff operation.

Acceptance/verification:

- A minimal custom method can be implemented and used directly and in a history
  using public APIs only, if public extension is selected. The example must not
  import `_PowerProblem`, `_advance`, or private preparation helpers.
- Verify identity, boundary-only, and hold histories, along with rectangular/
  explicit-row rejection where required by a method's representation.
- Retain HiJAX sensitivity contracts, existing failure reporting, factor reuse,
  and higher-order support/explicit rejection. Check overhead if preparation or
  numerical execution paths change; an API move alone is not a speedup claim.
- Run `test_modular_methods.py`, `test_epochs.py`, `test_epoch_validation.py`,
  `test_power_action.py`, `test_krylov_frechet.py`, `test_krylov_derivatives.py`,
  `test_polynomial_krylov.py`, `test_gmres_krylov.py`, and affected history tests.

## Verification and completion

All commands run from `/Users/nicholas/Projects/drifft` in the existing project
environment. Use focused tests per milestone; update its docstrings/docs and
run lint/type checks before marking it verified. Reuse existing independent
oracles and add regression cases only for newly exposed contracts/failure cases.

```sh
.venv/bin/pytest -q tests/<affected_file>.py
.venv/bin/ruff check src tests benchmarks
.venv/bin/ruff format --check src tests benchmarks
.venv/bin/ty check src tests benchmarks
git diff --check
```

After all milestones, run the full suite once, build docs, and build a wheel:

```sh
.venv/bin/pytest -q --durations=15
.venv/bin/zensical build
uv build --out-dir /private/tmp/drifft-api-consistency-dist
```

Check public imports from the built wheel, final rendered signatures, maintained
benchmark callers, and the canonical end-to-end examples. Do not rewrite
historical benchmark measurements to imply they used the changed API. Keep
documentation focused on the current interface; no migration page or
compatibility entrypoints are required.

Baseline evidence: at `c82f213`, the full Lineax suite passed 605 tests with 10
skips in 338.02 seconds. The subsequent review ran 29 focused API/transition/
occupation tests successfully and reproduced the additive coefficient scaling.
These results describe the starting code, not future milestones.

## Resume record

Milestone 1: implemented `ExpectedVisitsStatus`, separate `solver_status`, and
default throwing after the complete finite-output/residual acceptance check.
Updated API/guide/migration docs and benchmark paths that retain failed
candidates. The underlying solver, tolerances and recurrence are unchanged.
Verification: 90 targeted occupation/transition/validation/backend/derivative/
benchmark cases passed in 64.38 seconds; Ruff and ty passed. A controlled
solver-success regression covers residual rejection and NaN/inf under JIT/vmap.
Direct review found no unresolved milestone-1 issue.

Milestone 2: additive selection now stores `s` and derives `(s/2, s)`, including
Jacobi assembly. Updated docs/examples and regenerated both affected SVG figures
using the actual example scripts. Example fitnesses are preserved by doubling
their inputs; the sensitivity example verifies finite differences in new units.
Verification: 195 selection/Jacobi/dense/geometry/status tests passed in 103.30s.
Ruff and ty passed. Independent genotype-survival and gradient references cover
negative, zero and positive coefficients. No unresolved direct-review finding.

Milestone 3: demographic factories now have explicit `_factory` names;
`Transition(backend, probabilities=...)` is the sole explicit-row constructor.
Migrated maintained Python/docs/benchmark callers without aliases. A construction
count regression confirms shared factory reuse for repeated population sizes.
Verification: 75 transition/history/custom-method/dense/benchmark cases passed,
6 optional-dependency skips in 57.97s. Ruff/ty passed; direct review found no
unresolved construction issue.

Milestone 4: removed segregating suffix methods and migrated operations to the
user-selected `segregating=True` flag. Demographic singleton input is now
`singleton_mutation_rate`, rejected in full-state mode. The one-step advance
remains the exact transpose of the selected full/interior expectation map.
Verification: 173 transition/history/epoch/validation/custom-method/dense/Krylov
gradient cases passed in 119.36s. Ruff/ty passed; direct review checked source
chronology, flag validation, and removal of maintained suffix references.

Milestone 5: removed FFT/NUFFT dtype arguments; all four backends reject disabled
x64 before allocation. Typed float64 converters retain dynamic array leaves
and generate `float | jax.Array` constructor inputs. Epoch backend annotations
now use `AbstractTransitionBackend`. Verification: 129 API/validation/frequency/
backend cases passed in 26.84s; Ruff, ty and docs build passed (2.29s). Runtime
signature inspection confirmed the intended input types and removed dtype knobs.

Current state: all six milestones verified; implementation complete.
Chosen design: expose a read-only `EpochProblem` protocol and public
`prepare_epoch()`; retain current epoch backend/map arguments and the runtime
`_PowerProblem` representation. The protocol documents the stable algorithm
inputs and reconstruction/result checks; it does not expose projection metadata
or promise support for constructing arbitrary user problem objects. Keep matched
expectations Sequential-only and custom derivative responsibilities explicit.
Milestone-6 implementation: public preparation is now the shared path for direct
and demographic calls. Custom-method tests implement their own loop and finish
with `problem.result`, using public imports only. The runtime representation is
unchanged; `EpochProblem` is a typing protocol rather than a wrapper allocation.
The result helper checks state/scalar shapes and invalid error/tolerance inputs;
numerical failures retain the common `throw`/status behavior. Matched demographic
expectations remain explicitly Sequential-only.

Milestone verification: 214 extension/epoch/power/Krylov derivative cases passed in
174.25s. Ruff, ty and docs build passed. Executed all Python blocks in the
quickstart and five core guide pages successfully. Rendered signatures show
public types with no `_PowerProblem` leak.

Final integration:

- Full suite, including the final input guards: **621 passed, 10 skipped in
  352.63 seconds**. Log: `/private/tmp/drifft-api-full-tests.log`.
- Ruff lint/format, ty and whitespace checks passed.
- Final docs build passed; rendered signatures and glossary agree with the API.
  Log: `/private/tmp/drifft-api-docs-final.log`.
- Wheel and source distribution built in
  `/private/tmp/drifft-api-consistency-dist`. Isolated Python imported directly
  from the wheel and verified transition/history/preparation/occupation calls
  and removal of the former convenience entrypoints.
- Executed all Python blocks in the quickstart and five core guides. Executed
  and regenerated both affected plotting examples, including finite-difference
  checks for the rescaled additive coefficient.
- Final direct review covered public imports/signatures, failure/status
  precedence, coefficient units, source chronology, PyTree/AD boundaries,
  maintained callers, and the untracked plan. No unresolved in-scope finding.
- No new source modules, compatibility aliases, commits or pushes. The changes
  are uncommitted on `main`; unrelated untracked research artifacts are preserved.

Implementation is complete. User subsequently authorized a local commit of the
refactor and follow-up hygiene; no push is authorized.

After every milestone record actual changes, accepted contract decisions, tests
run/results, unresolved findings, and the exact next step. Before resuming after
compaction, read this plan, inspect git status/recent commits, and reconcile work
already present. Do not repeat verified milestones or silently change proposals
into accepted decisions. Commit or push only when subsequently requested.

## Follow-up hygiene — verified

User requested a hygiene pass and explicitly removed the need for API migration
documentation or backward compatibility. Kept `backend_factory` and
`boundary_backend_factory` after the naming discussion.

- Removed `docs/migration.md` and historical compatibility-module prose from the
  contributor guide. Current guides describe the supported API directly.
- Routed Sequential through the same preparation/solve orchestration as the
  Krylov methods. Consolidated demographic boundary application and clarified
  the private singleton-source parameter name.
- Aligned API mathematical docstrings with the glossary: observables `f`,
  population mass `v`, offspring map `q`, visits `u`, working exponent `k`,
  Jacobi coordinate/operator and angular/fitness-ratio notation.
- Added the missing static-bool guard on `as_lineax`; a traced flag previously
  produced `TracerBoolConversionError`, and now consistently raises the public
  `ValueError` across expectation, advancement and operator conversion.
- Improved shape/failure messages and expected-visits status documentation;
  removed a vacuous all-zero error-norm assertion.
- Fixed the README's stale epoch link and expected-visits instructions. Checked
  all local README/docs Markdown links and executed README plus four core guides.

Verification: 131 focused transition/operator/epoch/history/solve cases passed in
76.95s. The 11 public-API cases passed again after final readability edits.
All applicable pre-commit hooks (including Ruff/format/ty, AST, XML, JSON and
whitespace) passed. Final docs build passed in 1.65s, with no migration page
remaining in generated output. Full-suite results above predate this follow-up;
the changed runtime paths were verified with the focused suite.

Cleanup stays in existing modules. User authorized committing these changes on
main; unrelated untracked research artifacts remain excluded. No push is
requested and no implementation work remains for this pass.
