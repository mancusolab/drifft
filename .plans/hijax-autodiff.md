# Unify custom autodiff on HiJAX

User direction: use HiJAX for drifft-owned custom differentiation. Implementation
is complete on `dev`; public APIs, numerical formulas and failure behavior are preserved.

- [x] Inventory: optimized backend kernels already use HiPrim on JAX 0.11.2.
- [x] Migrate selection JVPs and the raw Krylov derivative guard.
- [x] Replace Equinox primitive registration and cached custom JVP in `_power.py`
  with HiPrim power and Frechet operations.
- [x] Validate mixed operator PyTrees, symbolic zeros, JIT, mapped derivatives,
  higher Sequential derivatives and first-order Krylov failure contracts.
- [x] Update autodiff documentation and review the complete diff.
- [x] Run relevant tests and repository checks.

Design: each operation owns its primal, analytic JVP, derived linearization/VJP
and batching rule. Linear arguments additionally have explicit transpose rules.
Equinox partition/combine preserves static Python leaves at the operator boundary;
Lineax tangent operators retain None for symbolic zero directions. Dense stays
ordinary JAX as an independent reference. No change to sensitivity tolerances,
Krylov adjoint mathematics, or higher-order support.

Evidence so far: frequency-map and Krylov derivative tests: 125 passed. Initial
operator tests exposed HiJAX Zero as a registered PyTree node; explicitly treating
AD sentinels as leaves fixes conversion to Lineax's filtered tangent representation.
Operator regression verification passed.

Reference: JAX custom derivative documentation
https://docs.jax.dev/en/latest/301/custom-derivatives.html
API checked against the installed JAX 0.11.2 source.

Skill checklist scope: no new data ingestion, PRNG, solver algorithm or Equinox
module hierarchy. HiJAX replaces the skill's Equinox custom-primitive preference
at the user's explicit request. Existing dtype and runtime failure contracts stay.

Review and validation update:
- Reviewed the changed primitives and their callers directly; no remaining
  actionable findings identified.
- Focused suites: 248 passed (125 frequency/guard, 46 Frechet, 73
  evolution/Dense/Jacobi, plus four new transformation tests).
- Added dense-reference regression coverage for reusable linearization with
  mapped tangent directions, jacfwd/jacrev, mapped gradients, and ignoring cached
  value tangents.
- Ruff lint/format, ty and diff whitespace checks pass.
- Final full suite: 636 passed, 10 skipped in 396.07 seconds. Optional dependency
  tests account for the skips. This run includes the final typing fixes.
- The initial full run was interrupted by a server restart; the completed rerun
  is logged at `/private/tmp/drifft-hijax-pytest.log`.
