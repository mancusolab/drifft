# Local FFT diagnostic validation

The implementation adds `FFT.diagnostics()` and an analytic value contribution
to `FFT.error_report`. It preserves construction, interpolation, and application;
`grad_atol` planning remains unsupported. The total floating-point error remains
unknown. The mathematical source is `drifft_local_fft_sobolev_analysis.md`,
particularly sections 3–5 and the numerical-stability discussion.

The diagnostic uses exact rational derivative-energy polynomials and Bernstein
range enclosures, the conservative Taylor/Peano derivative constant (4.6a),
signed interpolation-weight bounds, Chernoff tail bounds, and outward Decimal
log/exp enclosures. Direct derivative tails use the smaller of the
Cauchy–Schwarz bound and the telescoping boundary-PMF bound when applicable.
It does not implement automatic geometry search, smoothing, or a finite-precision
certificate. Bounds describe exact binomial nodes at the stored grid coordinates.

## Automated checks

Run from the repository root:

```sh
.venv/bin/python -m pytest -p no:capture tests/test_fft_bounds.py -q
```

All 31 tests passed. They check energy polynomials against independently
constructed binomial score derivatives, interpolation constants, polynomial
range enclosures, binomial tails, exact-node interpolants, interface jumps,
stencil derivative jumps, direct-only reflection, balancing invariance, and
actual JAX rows/JVPs/VJPs/forward propagation. SciPy binomial PMFs provide the
independent row references. Empirical floating-point allowances in tests are
not included in the analytic report.

## Actual JAX stress measurements

Environment: macOS 26.4.1 ARM64 CPU, JAX/jaxlib 0.11.2, NumPy 2.5.4,
SciPy 1.18.1, x64 enabled. Each case evaluates 401 equally spaced probabilities
in [0,1], with three observables bounded by one: constant ones, alternating
signs, and `default_rng(42).uniform(-1,1,M+1)`. The measured error is the maximum
absolute expectation difference from SciPy, not a maximum row-L1 norm or a
uniform certificate. All other constructor settings retain their defaults.

| M | min_mode_count | interp_order | Measured error | Analytic value bound | FFT amplification estimate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 8 | 8 | 6.189e5 | 2.143e-3 | 2.424e23 |
| 512 | 16 | 16 | 9.796e3 | 8.619e-6 | 1.056e22 |
| 512 | 128 | 12 | 9.456e-13 | 2.077e-11 | 4.456e2 |
| 1024 | 128 | 12 | 7.363e-13 | 2.285e-11 | 3.002e3 |
| 2048 | 128 | 12 | 5.921e-13 | 2.593e-11 | 8.891e3 |

The two small-cutoff cases demonstrate why analytic and arithmetic errors must
remain separate. Their small exact-arithmetic approximation errors do not
prevent catastrophic cancellation in the actual FFT calculation. These
measurements are platform-specific and are not asserted as portable regression
thresholds. The tests instead verify the amplification diagnostic and compare
stable default configurations with independent references.

## Repository verification

- Full regression invocation, `.venv/bin/python -m pytest -p no:capture -q`:
  1,050 passed in 703.20 seconds. It collected before the last six diagnostic
  tests were added; the final diagnostic module was separately rerun with all
  31 tests passing. Two warnings came from existing Krylov tests deliberately
  supplying array-valued static configuration.
- Ruff checks for the changed Python files: passed.
- `.venv/bin/ty check`: passed.
- `.venv/bin/zensical build --strict`: passed, no issues.
