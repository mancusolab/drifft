# Evolution engineering refactor

Scope: preserve the pending `backends` → `_backend` migration and implement the
four agreed review recommendations. No compatibility aliases, commits, or pushes.

1. [x] Declare representation requirements on evolution methods; use them in
   direct preparation and demographic validation. Regress custom-method behavior.
2. [x] Declare internal power/AD contracts; call sensitivity kernels directly
   instead of dispatching through pass-through method callbacks.
3. [x] Share rational Arnoldi construction and dense shifted-solve preparation,
   preserving GMRES diagnostics and polynomial image reuse.
4. [x] Move Jacobi reduced-operator assembly behind a private backend operation.
5. [x] Run focused numerical/AD tests, full suite, lint, formatting, types, docs,
   and review the resulting changes.

Baseline: 583 tests passed, 10 optional tests skipped after backend rename;
Ruff, formatting, ty, and strict docs build passed.

Progress: custom-method/domain tests passed (22); power/AD tests passed (90).
Rational/GMRES sensitivity checks passed (101). Ruff, formatting, ty, and
strict docs passed. Full-suite verification and final review are running.

Final verification: 586 passed, 10 optional tests skipped in 354.43 s.
Ruff, formatting, ty, diff checks, and strict docs build passed. Direct
review confirmed preserved source timing, polynomial image reuse, residual
checks, and the cached-primal derivative boundary. Changes remain uncommitted.
