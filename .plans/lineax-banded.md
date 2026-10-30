# Lineax banded-solver prototype

Branch: `lineax`, based on 99af547. Preserve the current banded Cholesky scans,
linear storage/work, one factorization per Jacobi assembly, and higher-order AD.
No public exports or compatibility aliases. User authorized committing the
implementation and merging into local main; no push is requested.

Decision (2026-10-30): user accepted the implementation and measured performance
tradeoff. Retain the Lineax solver and integrate it into local main.

- [x] Add private symmetric pentadiagonal operator, factor state, and explicit
  Lineax Cholesky solver in `src/drifft/_banded.py`.
- [x] Migrate Jacobi assembly to init once and reuse state through linear_solve.
- [x] Test independent dense values, JVP/VJP, higher derivatives, batching,
  factor-state reuse, failure reporting, and absence of dense materialization.
- [x] Compare complete assembly compilation, warm values/gradients, and compiled
  memory against the original implementation captured in /private/tmp.
- [x] Run relevant integration checks and report measured tradeoffs before
  deciding whether the prototype should be retained.

Verification: the full suite passed, 605 passed and 10 skipped in 338.02 seconds,
including 19 new banded cases. Separately, 74 Jacobi/likelihood integration cases
passed. Ruff, formatting, ty, and whitespace checks passed.

## Performance comparison

Compared the complete original and migrated diploid Jacobi assembly in separate
CPU float64 processes, with compilation caching disabled and five synchronized
warm executions per operation. Inputs combine diploid selection and mutation.
Temporary memory below is XLA's compiled temporary allocation, not peak RSS.

| Geometry | Operation | Original warm time | Lineax warm time |
| --- | --- | --- | --- |
| M=128, rank=32, padding=96 | Assembly | 0.214 ms | 0.239 ms |
| M=128, rank=32, padding=96 | Value and gradient | 0.638 ms | 0.533 ms |
| M=512, rank=197, padding=444 | Assembly | 3.024 ms | 2.989 ms |
| M=512, rank=197, padding=444 | Value and gradient | 7.883 ms | 7.557 ms |
| M=28948, rank=256, padding=640 | Assembly | 4.923 ms | 5.287 ms |
| M=28948, rank=256, padding=640 | Value and gradient | 15.353 ms | 14.183 ms |
| M=28948, rank=1582, padding=3184 | Assembly | 144.124 ms | 149.191 ms |
| M=28948, rank=1582, padding=3184 | Value and gradient | 354.325 ms | 339.471 ms |

At the default human-scale geometry, the migrated assembly is 3.5% slower;
value-and-gradient is 4.2% faster and its temporary allocation decreases from
405.1 MB to 363.3 MB (10.3%). Compilation changes from 0.287 to 0.356 seconds for
assembly and from 1.329 to 1.296 seconds for value-and-gradient.

All four assembled matrices matched bitwise. The largest absolute gradient
difference was 8.9e-10; all gradients passed atol=2e-8, rtol=2e-9 comparisons.
These results demonstrate a modest AD benefit, not zero forward overhead.

Reproduction artifacts for this local experiment are in /private/tmp:
`drifft_banded_bench.py`, `drifft_jacobi_before_lineax.py`, and the
`drifft_banded_{baseline,lineax}{,_human}.{json,npz,log}` files.
