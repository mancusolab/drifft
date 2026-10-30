# Test runtime investigation

Preserve all numerical regimes, assertions, and transformation coverage. Keep
CI/default pytest as the full suite and provide an explicit development selection.
No persistent compilation cache or tolerance changes are planned.

- [x] Profile full suite with durations and JUnit timings on this machine.
- [x] Identify repeated compilation work in the measured hot paths.
- [x] Reuse stable compiled test callables where inputs share a contract.
- [x] Measure affected tests before/after in fresh processes.
- [x] Run the numerical development selection and required quality checks.

Pending engineering changes and the `_backend` migration remain uncommitted;
benchmark artifacts already supplied by the user remain untouched.

Profile: full suite 586 passed, 10 skipped in 360.42 s. Benchmark files
accounted for 36.202 s; three marked large-population cases for 3.840 s.
Largest module: GMRES/Krylov, 45.46 s across 18 cases.

Experiment: share a compiled independent dense demographic JVP/VJP oracle
across six existing cases in three modules. Fresh-process timings with
persistent compilation cache disabled: 36.56 s before, 34.88 s after.
This is a modest single-run difference, not a demonstrated suite-wide speedup.
An additional whole-JVP JIT wrapper measured 34.89 s and was reverted.

The numerical-only development selection passed: 555 tests, 41 deselected,
306.70 s. Ruff, formatting, ty, diff checks, and strict docs build passed.
CI/default test selection and all test cases/tolerances remain unchanged.

The full-profile baseline and development selection are different workloads;
the 53.72 s difference is not an estimate of a full-suite optimization gain.
All changes remain uncommitted.
