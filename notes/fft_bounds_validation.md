# Local FFT diagnostic validation

This records the diagnostic checkpoint at commit `0ad5f7c`. Its API and test
counts are historical. The subsequent simplification integrates value bounds
into `fft.py`, removes the separate diagnostic API, and uses `atol` to plan
interpolation and sampling. For the current value/gradient interface and
two-grid construction, see the [FFT method documentation](../docs/methods/fft.md)
and [architecture investigation](fft_architecture_investigation.md).

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

## Follow-up: unified value planning

Value-bound calculations now live in `src/drifft/_backend/fft.py`. The separate
module, three diagnostic record types, and `diagnostics()` method are removed.
Preparation stores `interpolation_bound` and `sampling_bound`; `error_report`
reads them using the shared backend report without host-array conversion.

With `interp_order=None`, preparation tries orders 12, 14, and 16, then enlarges
the direct-region cutoff if needed, ending at direct-only evaluation. Each
analytic contribution has budget `atol / 4`. Explicit interpolation orders
preserve the requested geometry and can exceed the interpolation budget,
consistent with Jacobi's explicit rank override.

Exact polynomial enclosures remain for interpolation. A common Lebesgue bound
of two replaces order-specific weight diagnostics; the tests verify it on all
supported orders using exact rational polynomial enclosures, not a sampled
maximum. Sampling bounds use the existing SciPy binomial tail calculations,
and retained windows expand until the computed tail target is met. Floating-
point evaluation is still uncertified. Derivative/jump diagnostics remain in
the checkpoint; `grad_atol` planning remains unsupported.

Validation of the simplified implementation:

- FFT bounds, shared accuracy, FFT derivatives, spectral and validation modules:
  131 passed, including actual JAX rows/JVPs/VJPs/transposes for default and
  tight tolerance plans.
- Ruff lint/format checks and whole-project type checking passed.
- Strict documentation build passed.
- Demography, epochs, and demographic autodiff integration modules: 164 passed.

## Follow-up: joint branchwise gradient planning

FFT now accepts `grad_atol` with `angular_sensitivity`. Automatic construction
selects one geometry meeting both analytic value and branchwise parameter-
derivative allocations. An explicit interpolation order must meet the gradient
allocation or construction raises `ValueError`; its value override semantics
are unchanged. No accuracy checks were added to application kernels.

The interpolation derivative uses equation (4.7), reusing the halo energy
calculation at orders m and m+1. Angular conversion uses the halo's minimum
variance, a conservative enclosure of the core. Node-tail amplification uses
a uniform bound of four on the sum of absolute derivatives of the central
Lagrange weights, verified with rational polynomial enclosures for all allowed
orders. Direct-boundary tails use the score identity and an analytic maximum,
with the Cauchy–Schwarz fallback when the retained window ends below the mean.
Quantiles below floating-point resolution are handled by starting at a
representable quantile and expanding the retained window against the tail
bound. Unrepresentable planning budgets raise errors.

`interpolation_grad_bound` and `sampling_grad_bound` are stored in parameter
units. The existing report labels `grad_coordinate` as branchwise. Value jumps
and numerical arithmetic remain outside this gradient promise; the total
floating-point gradient bound remains unknown. No diagnostic classes or new
public reporting interface were introduced.

Verification:

- FFT bounds, backend accuracy, FFT derivatives, spectral, validation, epochs,
  and demographic autodiff modules: 247 passed.
- Four additional coarse-order derivative-bound cases: 4 passed.
- Independent checks cover exact-node binomial interpolants, both sides of
  switches, direct-tail maxima, JIT JVP/VJP/transpose derivatives, zero
  sensitivity, strict targets, explicit-order conflicts, and underflow failure.
- Ruff lint/format checks, whole-project type checking, strict documentation
  build, and diff whitespace checks passed.

## Hygiene and numerical sanity pass

Two failure-handling issues were reproduced and fixed:

- The bound evaluator inherited the active Decimal context. Enabling its
  `Inexact` trap caused FFT construction to fail. Enclosures now use an isolated
  context; a regression test checks construction under altered precision,
  rounding, and traps without changing the caller's context.
- A NaN from a tail routine could terminate window expansion and be hidden by
  `max(0, NaN)`, yielding an apparent zero bound. Fault-injection tests reproduced
  this for both interior and direct windows. Shared tail evaluation now rejects
  invalid probabilities and handles full support analytically.

Planner geometry and bounds are unpacked explicitly instead of selecting
negative offsets in an eight-element tuple. The full-row derivative test passes
its identity matrix as a JIT argument to avoid folding a large padded constant.

Checks: 169 backend/accuracy/derivative/spectral/validation tests passed; 23
planner/report cases were rerun after the final unpacking cleanup and passed.
Ruff lint/format checks and whole-project type checking passed.

A CPU spot check used M=1024, 101 uniform queries, seed-4 uniform signed values
bounded by one, and angular tangents sqrt(q(1-q)). Default settings gave maximum
value error about 1.6e-14 and angular derivative error about 6.8e-12. A requested
`grad_atol=1e-14` selected direct-only geometry but measured derivative error
remained about 3.8e-12. These are sampled measurements, not certificates: the
analytic target still does not include floating-point error. The previously
recorded small-cutoff cancellation examples remain a numerical limitation.
