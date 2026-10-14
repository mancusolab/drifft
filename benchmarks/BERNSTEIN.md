# Chebyshev/NUFFT replacement evaluation

Measured on 2026-10-14, macOS ARM64 CPU, Python 3.12.13, JAX/JAXlib 0.11.2,
Equinox 0.13.8 and Lineax 0.1.1. FINUFFT uses upstream `jax-finufft` main,
commit `8ae08bb11248b0a61c0c1e2ddc72e02741a5e7f7`, built locally. Its published
1.3.1 release does not support this JAX version's batching API. Python 3.11
support was dropped for this branch. CUDA was not available or tested.

## Outcome

The replacement preserves the `Transition` API and greatly reduces stored plan
size and XLA-reported temporary memory. It is **slower in warm CPU applications
than the previous backend** in this initial comparison. This is an experimental
replacement, not evidence that it should replace the previous method in a release.
No dual-backend class hierarchy is shipped; the previous implementation is
preserved at Git revision `b13114f`.

Both revisions ran sequentially in isolated processes using the same interpreter,
JAX version, seeded signed vectors, uniform masses and compilation convention:
plan arrays are dynamic arguments, metadata are static. Timings are medians of
five synchronized warm calls. Each implementation uses its default settings;
both pass a sampled normalized forward-error threshold of `1e-9`. This is not a
proof of equal worst-case operator error or optimal tuning for either method.

At selection strength `1e-4`:

| Gene copies | Previous expect | New expect | Previous advance | New advance |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.587 ms | 1.419 ms | 0.798 ms | 1.593 ms |
| 10,000 | 2.157 ms | 10.081 ms | 5.096 ms | 10.674 ms |
| 28,948 | 4.846 ms | 25.268 ms | 13.517 ms | 28.305 ms |
| 100,000 | 17.713 ms | 64.881 ms | 48.591 ms | 73.330 ms |

For 28,948 gene copies, cached plan arrays decrease from **18,370,656 bytes to
10,360 bytes**. Reported XLA temporary storage decreases from 23,587,176 to
1,276,976 bytes for `expect`, and from 66,457,656 to 1,189,616 bytes for `advance`.
Compile times in that selected case are 0.217 versus 0.117 seconds for `expect`,
and 0.343 versus 0.129 seconds for `advance`. These are compilation times, not
fresh-process startup measurements. The raw results also include neutral selection
and every warm sample; one baseline neutral case was substantially slower than its
weak-selection counterpart, illustrating timing variability.

Source: [revision comparison](bernstein_revisions_cpu.json).

## Larger populations, derivatives and solves

The replacement-only sweep reaches one million gene copies. With `s=1e-4`, warm
`expect` takes **506.6 ms** and `advance` **570.1 ms**. The degree is 7,830 and the
padded stencil width 7,485. The largest sampled normalized value error in this
sweep is **6.6e-12**. Uniform-input mass defects stay below `7e-16`.

Derivative errors are a separate result: the maximum sampled absolute
selection-JVP error grows to **9.3e-8** at one million copies. At 28,948 copies it
is about `1.2e-10`; the optional larger-degree setting changes that to about
`9.7e-11` and increases runtime. This small improvement is not a derivative
certificate. Roundoff and conditioning can dominate despite a generous degree.
The reference uses the finite Bernstein derivative, not differentiation of a
singular angular coordinate.

The rectangular experiment uses 1,000 parental and 10,000 offspring gene copies,
with the larger derivative degree. Sampled value error is below `7e-15` and
selection-JVP error below `3.6e-11`. Multiple RHS are supported through `vmap` and
tested numerically; this study does not claim a specialized batched throughput win.

At 256 gene copies, both neutral and weak-selection expected-visits solves
converge. Maximum error against a dense solve is below `1.3e-13`. Warm solve
times are approximately 63–106 ms in this run. Tests independently compare small
expected-visits gradients with differentiated dense systems. Large population
expected-visits accuracy and gradients remain unvalidated: one-step accuracy
alone does not bound the inverse transient system's error.

Sources: [scaling](bernstein_cpu.json), [derivative degree](bernstein_derivative_cpu.json),
[rectangular](bernstein_rectangular_cpu.json), [small solves](bernstein_solve_cpu.json).

## fastDTWF comparison

The updated comparator runs drifft and PyTorch/fastDTWF in separate processes.
This avoids conflicting bundled OpenMP runtimes and retains exactly shared
probabilities and input distributions. JSON serialization is outside timing.

At 28,948 gene copies, `s=1e-4`, and fastDTWF `tv_sd=0.1`, the replacement takes
about **29 ms** per generation versus **191–197 ms** for fastDTWF across singleton,
uniform and seeded random masses. Between-method L1 differences are approximately
`1.0e-11`, `4.4e-6`, and `1.9e-5`, respectively. These are discrepancies, not exact
errors. This configuration-specific comparison does not establish equal accuracy
or a general speed advantage over fastDTWF. The previous backend is still faster
than this replacement on the matched warm revision comparison above.

Source: [isolated fastDTWF comparison](bernstein_fastdtwf_cpu.json).

## Memory and numerical qualifications

The plan caches only a node vector. Sampling regenerates one batch of relative
binomial weights at a time. Its explicit transpose scatters directly into one
output accumulator. HLO tests guard against a full population-sized zero-fill
per node batch. The cosine derivative recurrence uses parity suffix sums rather
than updating an entire coefficient vector once per degree.

XLA reports about 40 MB of advance temporaries at one million gene copies.
**This excludes FINUFFT-managed scratch.** The sweep's process RSS high-water mark
also includes JIT compilation and independent reference matrices, so it is not a
measurement of a single production matvec's peak memory. No device profiler or
CUDA memory claim is provided.

For the tail stencil, Bernstein's inequality bounds two-sided omitted mass by
`2 exp(-t² / (2*(variance+t/3)))`. With `A=log(2)-log(delta)` and
`t=sqrt(2*variance*A)+2*A/3`, the exponent is at most `-A`. The implementation
uses `variance <= M/4`, rounds upward, and adds one to cover the difference
between the binomial mode and mean. Clipping indices is accompanied by a mask;
reflection preserves the binomial tail bound. Normalizing retained weights adds
at most `2*delta*max(abs(x))` node error, then the conservative Lobatto Lebesgue
factor accounts for interpolation amplification.

The spectral bound concerns values, not derivatives. NUFFT tolerance is a
requested accuracy setting, not a rigorous norm bound. Endpoint rows are exact;
small negative interpolated entries elsewhere are reported and never clipped.
Sampling nodes are fixed, nontrainable geometry. Hold the plan fixed when taking
gradients of selection parameters.

## Cold startup and practical reuse at pre-expansion size

Measured at constant diploid effective size **14,474** (**28,948 gene copies**),
using the same pre-expansion scenario as the earlier fastDTWF report. Each case
uses three fresh interpreters and ten synchronized warm calls per interpreter.
Workers run sequentially, reversing order on alternating trials. Cold time is
launch to first result, including imports, device initialization, plan setup,
and compilation. JAX persistent compilation caching is disabled. OS filesystem
and dependency disk caches are not cleared; this is fresh-process latency, not
a first-install measurement. Warm time is the median of per-process medians.

| Selection | Implementation | Cold first result | Warm call | Estimated 10 calls | Estimated 100 calls |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Chebyshev/NUFFT | 0.805 s | 27.59 ms | 1.053 s | 3.536 s |
| 0 | Previous FFT | 3.790 s | 13.11 ms | 3.908 s | 5.088 s |
| 0 | fastDTWF (`tv_sd=0.1`) | 2.432 s | 178.27 ms | 4.036 s | 20.081 s |
| 0 | fastDTWF (`tv_sd=0.01`) | 3.249 s | 1118.87 ms | 13.319 s | 114.017 s |
| 0.0001 | Chebyshev/NUFFT | 0.861 s | 29.24 ms | 1.124 s | 3.756 s |
| 0.0001 | Previous FFT | 3.852 s | 13.71 ms | 3.975 s | 5.209 s |
| 0.0001 | fastDTWF (`tv_sd=0.1`) | 2.456 s | 192.87 ms | 4.192 s | 21.550 s |
| 0.0001 | fastDTWF (`tv_sd=0.01`) | 3.478 s | 1202.91 ms | 14.304 s | 122.565 s |

The new backend saves about **3 seconds at startup**. Its plan preparation takes
about 0.20 seconds versus 2.39–2.42 seconds for the previous implementation.
Warm calls are about twice as slow. The first whole-call counts where the previous
backend becomes faster in estimated total runtime are **208 calls for neutral
selection and 194 calls for selection strength `1e-4`**.
This favors the replacement for short-lived jobs; the previous backend remains
faster for sufficiently long reuse of an existing plan. The new backend is
faster cold and warm than fastDTWF at both tested tolerance settings.

Totals use `cold + (calls - 1) * warm`; they are **estimates**, not timed evolving
population trajectories. Every warm call propagates the same uniform input.
Population size and selection remain fixed; no exponential expansion is modeled.
This does not measure gradients, solves, or changing-size histories. Tolerance
settings (`tail_tol=1e-12`, fastDTWF `row_eps=1e-8`) do not establish identical
approximation error; see the accuracy comparison elsewhere in this report.

The previous source was checked byte-for-byte against revision `b13114f`.
The current source is `bd024c7`; benchmark-only changes add source routing.
Package version metadata describes the shared installed environment; each
drifft worker separately records its actual imported source path. Raw samples:
[bernstein_startup_cpu.json](bernstein_startup_cpu.json).

```sh
uv run --group fastdtwf python benchmarks/compare_startup.py \
  --baseline-source .cache/bernstein-baseline/src --warm-repeats 10 \
  --output benchmarks/bernstein_startup_cpu.json
```

## Reproduction

```sh
uv sync --locked --group fastdtwf
mkdir -p .cache/bernstein-baseline
git archive b13114f src | tar -x -C .cache/bernstein-baseline
uv run python benchmarks/compare_revisions.py \
  --baseline-source .cache/bernstein-baseline/src \
  --output benchmarks/bernstein_revisions_cpu.json
uv run python benchmarks/benchmark.py --device cpu \
  --output benchmarks/bernstein_cpu.json
uv run python benchmarks/benchmark.py --device cpu --sizes 28948 \
  --derivative-accuracy --repeats 3 --output benchmarks/bernstein_derivative_cpu.json
uv run python benchmarks/benchmark.py --device cpu --sizes 10000 \
  --source-gene-copies 1000 --derivative-accuracy --repeats 3 \
  --output benchmarks/bernstein_rectangular_cpu.json
uv run python benchmarks/benchmark.py --device cpu --sizes 256 --repeats 3 \
  --output benchmarks/bernstein_solve_cpu.json
uv run --group fastdtwf python benchmarks/compare_fastdtwf.py \
  --sizes 28948 --selections 0 0.0001 --tv-sds 0.1 --repeats 3 \
  --output benchmarks/bernstein_fastdtwf_cpu.json
```
