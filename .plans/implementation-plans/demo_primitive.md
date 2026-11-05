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
