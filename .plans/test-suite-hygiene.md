# Test suite hygiene

Scope: audit and consolidate tests after the HiJAX migration. Preserve meaningful
numerical regimes and failure coverage; do not optimize for a target test count.

Inventory: 184 test functions produced 646 cases (636 passed, ten optional
fastDTWF-related skips). The revised suite has 182 functions and 589 cases.

Changes:
- Overdominance validation: replace 27 Cartesian combinations with nine cases
  covering every pair of order, invalid strength and generation regime. The
  shared validation predicate runs before each action path.
- Occupation solves: retain all selection regimes at size 16 and representative
  size-64/256 cases for both backends. Keep the separate truncated spectral solve.
- Remove repeated multi-step counts from exact small power tests; retain zero,
  one and repeated steps, plus separate long/truncated power tests.
- Stop repeating successful breakdown sensitivities for both throw settings;
  keep throw=False and dedicated failure-path tests.
- Replace conservation-only gradient objectives with nonconstant rewards and an
  analytic mutation-mean sensitivity (including a zero mutation rate).
- Move the dense evolution oracle to tests/_reference.py and make its frequency
  formula independent of production selection code.
- Fold keyword-call checks into transition-direction checks. Remove a standard
  NamedTuple roundtrip test; check the public result type in actual solve tests.
- Merge HiJAX transformation regressions into test_power_action.py.
- Mark benchmark integration tests explicitly. Default pytest and CI still run
  everything; the focused numerical selection has 548 cases, excluding 38
  benchmark cases and three already-marked large-population cases.

Kept: endpoint and higher-derivative regressions, small/full vs truncated geometry,
nonconvergence and failure diagnostics, cheap planner/validation boundary cases,
and benchmark process/protocol tests. These protect distinct contracts.

Verification: main affected suites passed 251 tests; shared-reference consumers
and final solve checks passed 83 tests (13 solve cases overlap). Final Ruff lint,
formatting, ty and diff-whitespace checks passed. Collection confirms 589 total
cases and 548 in the focused numerical selection. Benchmark markers only change
selection; their test bodies are unchanged. The pre-cleanup full run passed 636
with ten optional skips. No runtime improvement is claimed: previous full-suite
timing and current focused-suite timing are different workloads.
