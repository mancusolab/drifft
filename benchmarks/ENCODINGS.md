# NUFFT encoding comparison

Measured 2026-10-24 on macOS 26.3.1 ARM64 CPU, Python 3.12.11,
JAX/JAXlib 0.11.2, Equinox 0.13.8, NumPy 2.5.4 and SciPy 1.18.1.
`jax-finufft` is the locked revision `8ae08bb11248b0a61c0c1e2ddc72e02741a5e7f7`,
built locally using FFTW, with OpenMP disabled. JAX uses its default CPU threading.
GPU performance was not measured.

## Decision

Keep `encoding="lobatto"` as the default and offer `encoding="filtered"` as an
explicit option. Filtering provides the stronger spectral certificate described
in `wf_filtered_chebyshev_bounds.md`, but the measured benefit depends on size
and accuracy. It is not a universal speed or numerical-accuracy improvement.

The ordinary planner now also uses the note's sharper Lobatto bound. Much of
the mode-count reduction is available without changing the encoding. This
changes automatically chosen degrees, not the default interpolation algorithm.

The filtered implementation includes the global linear endpoint correction in
coefficient space, its factor-two spectral and sampling error allowances, and
an untapered exact-degree fallback. It remains a one-generation backend;
this experiment does not implement persistent reduced-state evolution or
Jacobi-style assembly and powering.

## Method

Each size, accuracy, encoding, operation and cold repetition runs in a fresh
process, with persistent compilation caching disabled. The parent times
launch-to-first-result. Workers separately report imports, setup, compilation,
first execution, and seven synchronized warm calls. Main results use three
processes per case and medians of the per-process warm medians; supplementary
experiments use two processes. Encoding order reverses on alternate trials.
Processes run sequentially, without concurrent test or numerical-review work.
These are process-cold measurements, not cold filesystem-cache measurements.

Timing inputs are a sinusoidal observable for `expect`, uniform mass for
`advance`, and underdominance strength 0.1. Geometry and input arrays are dynamic
compiled arguments. Diagnostics run after the timing samples. Native FINUFFT
workspace is excluded from XLA temporary-memory figures.

The `legacy` baseline uses the current execution machinery with only the old
`4M/d * exp(-d²/(2M))` degree planner restored inside benchmark construction.
It isolates planning changes; it is not a separate historical package build.

## Mode counts and warm costs

All degrees below are **output polynomial degrees**. Filtered encoding samples
one more node than ordinary interpolation at equal output degree.

| M | Accuracy | Legacy degree | Lobatto degree | Filtered degree | Lobatto expect (ms) | Filtered expect (ms) | Lobatto advance (ms) | Filtered advance (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1e-6 | 192 | 190 | 187 | 0.896 | 1.498 | 0.985 | 1.099 |
| 1,000 | 1e-10 | 234 | 233 | 230 | 1.428 | 1.346 | 1.478 | 1.459 |
| 10,000 | 1e-6 | 623 | 607 | 594 | 5.938 | 6.093 | 6.658 | 6.872 |
| 10,000 | 1e-10 | 754 | 743 | 733 | 8.657 | 8.538 | 9.864 | 9.624 |
| 100,000 | 1e-6 | 2,025 | 1,926 | 1,880 | 46.023 | 43.573 | 56.517 | 51.814 |
| 100,000 | 1e-10 | 2,430 | 2,357 | 2,319 | 60.434 | 61.973 | 71.384 | 70.768 |

At M=100,000 and accuracy 1e-6, filtered advance is about 8.3% faster than
sharper Lobatto and 12.0% faster than the legacy planner (58.906 ms).
At accuracy 1e-10 the filtered/Lobatto advance difference is only 0.9%, while
filtered expectations are slower. Small cases show appreciable timing noise;
raw samples include outliers. These measurements do not establish statistical
significance for small differences.

## Cold costs and memory

For M=100,000, accuracy 1e-6, `advance`:

| Component | Lobatto | Filtered |
| --- | ---: | ---: |
| Launch to first result, median | 1.476 s | 1.396 s |
| Imports, median | 0.938 s | 0.859 s |
| Setup, median | 0.206 s | 0.197 s |
| Compilation, median | 0.191 s | 0.224 s |
| First execution, median | 0.059 s | 0.054 s |
| Stored geometry | 15,416 bytes | 15,056 bytes |
| XLA temporary memory | 4,047,280 bytes | 4,047,664 bytes |

Component medians do not add to the median total; launch time also includes
interpreter and harness startup. Imports dominate cold costs, and filtering
does not establish a cold-start advantage. In this case the stricter filtered
sampling allocation increases stencil radius from 972 to 1,000, offsetting
some of the work saved by fewer sampling nodes.

## Numerical accuracy

Diagnostics compare complete rows against SciPy binomial PMFs at endpoints,
near-boundary probabilities (including 1/M and 2/M), and seeded interior points.
The derivative reference uses the exact lower-degree binomial identity.
The following are **maxima over the tested rows**, not continuum certificates.

| M | Accuracy | Lobatto row-l1 | Filtered row-l1 | Lobatto relative derivative-l1 | Filtered relative derivative-l1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1e-6 | 2.98e-8 | 5.07e-8 | 2.30e-7 | 9.06e-7 |
| 1,000 | 1e-10 | 2.75e-12 | 5.33e-12 | 2.41e-11 | 1.47e-10 |
| 10,000 | 1e-6 | 3.33e-8 | 6.08e-8 | 2.09e-7 | 9.47e-7 |
| 10,000 | 1e-10 | 3.21e-12 | 5.64e-12 | 3.12e-11 | 1.42e-10 |
| 100,000 | 1e-6 | 3.78e-8 | 6.19e-8 | 2.86e-7 | 9.66e-7 |
| 100,000 | 1e-10 | 2.22e-11 | 2.29e-11 | 3.62e-10 | 3.65e-10 |

Both encodings can produce negative entries. At accuracy 1e-6 the largest
sampled negative row mass is approximately 1.52e-8 for Lobatto and 2.55e-8 for
filtered. The implementation does not clip or renormalize outputs.
Neither the analytic value bound nor requested FINUFFT tolerance certifies
derivative accuracy, floating-point error, or relative accuracy in rare tails.

Dense power references at M=1,000 cover neutral and strength-0.1 evolution,
singleton and uniform initial distributions, and 1, 10 and 100 generations.
Worst final l1 errors at 100 generations are:

| Accuracy | Lobatto | Filtered |
| ---: | ---: | ---: |
| 1e-6 | 1.82e-11 | 2.33e-8 |
| 1e-10 | 1.17e-13 | 2.39e-12 |

These absolute errors are small, but the stronger worst-case one-step
certificate does not imply more accurate repeated evolution for these inputs.

## Equal-degree comparison

The optional `filtered-matched` benchmark variant forces the same output degree
as sharper Lobatto while retaining the filtered sampling policy. It introduces
no additional public encoding. At M=1,000 and accuracy 1e-6, both use degree 190:

| Metric | Lobatto | Filtered at equal degree |
| --- | ---: | ---: |
| Maximum sampled row-l1 | 2.98e-8 | 2.71e-8 |
| Maximum relative derivative-l1 | 2.30e-7 | 5.21e-7 |
| Worst 100-generation final l1 | 1.82e-11 | 1.13e-8 |
| Warm advance | 1.020 ms | 1.071 ms |

At equal degree the sampled row error improves slightly, but the derivative
and repeated-evolution tradeoffs remain. At M=10,000, degree 607, sampled row
error similarly drops from 3.33e-8 to 2.69e-8, while the maximum relative
derivative error increases from 2.09e-7 to 4.46e-7. This rules out reduced degree
as the sole explanation for the numerical differences in the main sweep.

## Larger-degree derivative heuristic

With `derivative_accuracy=True` and accuracy 1e-6:

| M | Encoding | Degree | Maximum sampled row-l1 | Maximum relative derivative-l1 | Warm advance (ms) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1,000 | Lobatto | 285 | 1.25e-11 | 5.34e-11 | 1.406 |
| 1,000 | Filtered | 281 | 1.24e-12 | 1.16e-11 | 1.480 |
| 10,000 | Lobatto | 911 | 9.18e-10 | 5.30e-9 | 10.807 |
| 10,000 | Filtered | 891 | 1.31e-10 | 1.26e-9 | 10.161 |

Here relative derivative-l1 means the l1 error in the probability derivative
row divided by the l1 norm of its exact derivative row. The larger-degree
heuristic substantially improves this metric in these cases, with extra runtime.
It still does not provide a derivative certificate. Increasing degree does not
eliminate binomial-stencil, NUFFT or arithmetic error; a tighter value accuracy
setting also changes sampling budgets.

## Reproduction and raw results

```sh
uv run python benchmarks/compare_encodings.py \
  --sizes 1000 10000 100000 --accuracies 1e-6 1e-10 \
  --cold-repeats 3 --warm-repeats 7 --diagnostics \
  --output benchmarks/encoding_cpu.json

uv run python benchmarks/compare_encodings.py \
  --sizes 1000 10000 --accuracies 1e-3 1e-6 \
  --variants lobatto filtered-matched --cold-repeats 2 --warm-repeats 7 \
  --diagnostics --output benchmarks/encoding_matched_cpu.json

uv run python benchmarks/compare_encodings.py \
  --sizes 1000 10000 --accuracies 1e-6 --variants lobatto filtered \
  --derivative-accuracy --cold-repeats 2 --warm-repeats 7 --diagnostics \
  --output benchmarks/encoding_derivative_cpu.json
```

Raw data: [main sweep](encoding_cpu.json),
[equal-degree sweep](encoding_matched_cpu.json),
[derivative-padding sweep](encoding_derivative_cpu.json).
