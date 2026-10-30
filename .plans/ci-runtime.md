# CI runtime

Repository/starting point: main at 545623a. User authorized optimizing CI and
GitHub Actions. Preserve every current test and both Python versions (3.12/3.13).
User subsequently authorized committing the completed CI changes locally.
No push, workflow dispatch, or remote configuration change is requested.
Preserve unrelated untracked research files.

## Evidence

Public GitHub API inspection succeeded without a login, although gh itself is
not authenticated. Latest successful checks run: 41012117786, ab9bc1d.

- Python 3.12: pytest 724s, uv sync 2s, pre-commit 7s, build 1s, FFTW install 8s.
- Python 3.13: pytest 489s, uv sync 1s, pre-commit 6s, build 1s, FFTW install 15s.
- Previous checks run 41007349029: pytest 733s/682s; dependency installs 86s/84s.
- Latest documentation run 41012117726 completed in 39s; dependency install 1s.
- Existing uv wheel caching is effective. The critical path is pytest, not docs
  or package installation. Retain the portable FINUFFT build settings.
- Earlier complete local JUnit profile totals 359s. Eight related evolution/
  history modules account for approximately 180s, leaving about 179s elsewhere.
  New cases and hosted-runner differences require measuring the resulting split.

## Changes

- [x] Add an opt-in, two-way whole-module pytest selector in tests/conftest.py.
  This is standard pytest collection configuration, not a new test framework or
  numerical module. Default pytest still runs everything. New modules enter
  shard 2 automatically; no maintained allowlist can silently drop new tests.
- [x] Run both shards on each Python version in separate hosted jobs. Preserve
  per-module JAX reuse and all benchmark/slow categories. Avoid within-runner
  worker oversubscription and additional test dependencies.
- [x] Run pre-commit/build once outside the test matrix; cache pre-commit's
  environments using the actual Python version and hook configuration.
- [x] Cancel superseded checks runs. An optional trigger-policy question was
  presented to the user; use PR updates/main pushes as the stated default if no
  answer arrives after sufficient opportunity. Retain manual workflow dispatch.
- [x] Upload per-shard JUnit reports and print durations for future CI profiling.
- [x] Leave Pages deployment unchanged: its measured duration is small and its
  current concurrency preserves in-progress deployments.
- [x] Document local reproduction commands and the wall-time/runner-minute tradeoff.

Persistent JAX compilation caching is deferred; the first optimization keeps
cold-compilation coverage and avoids introducing another cache to tune.

## Verification

1. actionlint and repository quality hooks.
2. Compare complete collection with both shard collections: disjoint, complete,
   and identical node IDs; include skipped optional cases in the comparison.
3. Execute both shards in fresh processes, sequentially on this machine, with
   synchronized wall times and JUnit results. Every collected test executes once.
4. Record measured shard durations; do not present a local max(shard durations)
   as an observed hosted parallel run. Hosted speedup requires a later push.
5. Review trigger/cancellation/cache scope, matrix coverage, and output paths.

Progress: implementation and local verification complete. Collection verified: 634 default cases =
232 in shard 1 + 402 in shard 2, with exact node-ID union and no overlap.
actionlint and ty passed; Ruff's import-spacing finding on the new conftest was
corrected. Both Python versions retain both shards and all existing categories.
The optional trigger question has not received a response after ample time;
the stated PR/main default is implemented, with workflow_dispatch for branches.
Both shards ran sequentially in fresh processes with persistent compilation
caching disabled:

| Shard | Result | pytest duration | Process wall time |
| --- | --- | --- | --- |
| 1 | 232 passed | 170.77s | 174.58s |
| 2 | 392 passed, 10 skipped | 180.25s | 183.02s |

Actual JUnit node IDs match the complete 634-case collection exactly once.
After tightening membership to exact test paths (so new nested files with an
existing basename still enter shard 2), recollection selected the identical
verified/executed node IDs. Total pytest work was 351.02s; the local longest
shard was 180.25s. The recent slightly earlier unsharded suite took 352.63s.
These are local sequential measurements, not an observed hosted parallel run.

Cache/artifact actions use the current documented releases (v6/v7), pinned to
tag commits verified with git ls-remote. Existing checkout/setup-uv/uv pins and
the effective portable wheel cache remain unchanged. Actionlint, Ruff, format,
ty, repository hooks and docs build passed. Public main branch metadata confirmed
there are no existing required check names to preserve. Final workflow review
checked permissions, cancellation groups, cache keys, artifact names, complete
Python/shard coverage, and unchanged Pages deployment.

No numerical code, tolerances, or assertions changed. User authorized a local
commit; no push or remote settings change is requested. Unrelated research
artifacts remain untouched. The first hosted run after a
subsequent push will provide actual CI timing and Python 3.13 verification.
