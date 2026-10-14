# Chebyshev/NUFFT replacement

The user's replacement instruction supersedes the opt-in/default-backend sections
of `codex_bernstein_chebyshev_nufft.md`. Keep one concrete plan and transition;
compare against git revision `b13114f`, rather than shipping two backends.

## Module responsibilities

- `_plan.py`: host-side static degree/error allocation and compact sampling geometry.
  Preserve `build_transition_plan(gene_copies=...)` and `TransitionPlan`; replace
  log-odds-specific tuning options with spectral, tail, NUFFT and batch controls.
- `_spectral.py`: pure batched mode-relative Bernstein sampling, DCT-I and
  endpoint-safe NUFFT evaluation. No cached node-by-stencil weight table.
- `_operator.py`: preserve `Transition` and `Underdominance`; replace only the
  expectation kernel. Retain automatic transposition, rectangular operations and
  Lineax conversion, subject to memory profiling.
- `_solve.py`: keep the existing matrix-free solver and result contract.
- Tests: preserve public/operator/solve contracts; replace obsolete log-odds
  geometry tests with sampling, spectral, NUFFT, endpoint AD and memory checks.
- Benchmarks: migrate configuration and diagnostics, preserve historical JSON,
  measure compile and warm execution separately and compare equal accuracy.

## Numerical decisions to validate

Static degree follows the supplied value-error bound. A separate derivative
setting increases degree conservatively; its accuracy is measured independently,
not certified by the value bound. Sampling uses conditional nonnegative binomial
weights, with a global sampling budget divided by a conservative Lobatto Lebesgue
bound. Endpoint rows are exact. A custom derivative evaluates the derivative
polynomial directly, avoiding angular singularities. No output clipping or mass
normalization. NUFFT tolerance is a requested numerical tolerance, not a rigorous
operator-norm certificate; roundoff remains separately assessed.

## Execution and verification

1. Install and smoke-test FINUFFT on the actual JAX/platform combination.
2. Write failing component and endpoint-AD tests, then replace the kernels.
3. Migrate tests, benchmarks and docs; inspect transposed executable memory.
4. Run numerical regression, type/lint, build and independent review.
5. Record measured accuracy/performance and limitations; do not infer a speedup.

Production code is Functional Core; benchmark scripts are Imperative Shell.
Core APIs remain vector-valued; multiple RHS use `vmap` and are tested explicitly.
CUDA validation is conditional on hardware availability.

## Resolved engineering decisions

- Sampling and cosine evaluation have matched primitive/JVP/transpose rules.
  Sampling transposes stream into one accumulator; parity suffix sums implement
  the coefficient derivative in linear work. Fixed sampling geometry is explicitly
  nontrainable, while input vectors and selection parameters are differentiable.
- `jax-finufft` follows upstream main, locked to `8ae08bb`; release 1.3.1 has a
  batching incompatibility with JAX 0.11.2. The source fix passes the core suite.
- Python 3.11 support is removed at the user's request; minimum Python is 3.12.
- fastDTWF comparisons isolate PyTorch and FINUFFT to avoid conflicting OpenMP
  runtimes. Reports retain the exact shared probability vectors and input masses.
- The replacement is slower in the initial CPU warm comparison, but has much
  smaller cached plans and XLA temporaries. See `benchmarks/BERNSTEIN.md` for the
  measured tradeoff; the old backend remains only in Git history.
