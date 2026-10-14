# Successive-generation CPU benchmark

This benchmark measures actual evolving distributions at fixed population size,
then an illustrative one-generation doubling in population size. It does not
reconstruct the Schiffels–Durbin demographic history.

## Results

Measured on 2026-10-14, macOS ARM64 CPU, Python 3.12.13 and JAX/JAXlib 0.11.2.
The production backend is revision `af1ec82`; the previous backend is archived
from `b13114f`. Dependency versions and imported source paths are in the JSON.
The only code additions are benchmark orchestration and tests.

**The replacement does not provide a general CPU speedup for long epochs.**
The previous FFT backend is faster once its setup is amortized. Condensed
fastDTWF is substantially faster than its ordinary propagation mode and can
also outperform the replacement, at a larger measured discrepancy from the
high-accuracy reference. Refreshing every 50 generations does not materially
improve accuracy for these uniform starting distributions.

### Epoch totals, selection strength 0

Seconds including per-epoch setup and JAX compilation/calibration; imports and
framework initialization excluded. These are the original single-run sweep.

| Implementation | 100 generations | 200 | 500 | 1,000 |
| --- | ---: | ---: | ---: | ---: |
| Chebyshev/NUFFT | 2.917 | 5.481 | 13.177 | 26.086 |
| Previous FFT | 3.525 | 4.204 | 6.287 | 9.859 |
| fastDTWF ordinary | 17.420 | 34.690 | 86.753 | 182.275 |
| fastDTWF condensed | 1.373 | 2.161 | 4.590 | 8.511 |
| fastDTWF refresh every 50 | 1.746 | 3.258 | 7.898 | 15.722 |

### Epoch totals, selection strength 0.0001

Seconds including per-epoch setup and JAX compilation/calibration; imports and
framework initialization excluded. These are the original single-run sweep.

| Implementation | 100 generations | 200 | 500 | 1,000 |
| --- | ---: | ---: | ---: | ---: |
| Chebyshev/NUFFT | 3.163 | 6.069 | 14.586 | 28.485 |
| Previous FFT | 3.579 | 4.283 | 6.418 | 10.040 |
| fastDTWF ordinary | 18.062 | 36.694 | 89.776 | 181.922 |
| fastDTWF condensed | 1.536 | 2.254 | 5.274 | 19.631 |
| fastDTWF refresh every 50 | 3.215 | 5.955 | 13.252 | 21.821 |

The selected condensed timings varied unusually near the end of the sweep;
additional fresh-worker measurements are reported below rather than replacing
those original observations. Do not interpret the neutral/selected timing
difference as a measured causal effect of selection.

### Accumulated discrepancy after 1,000 generations

Full-distribution L1 difference against the tighter reference (total variation
is half L1 for normalized distributions). This is not an exact-error guarantee.

| Implementation | Neutral L1 | Selected L1 |
| --- | ---: | ---: |
| Chebyshev/NUFFT | 1.269e-13 | 1.272e-13 |
| Previous FFT | 5.915e-11 | 5.885e-11 |
| fastDTWF ordinary | 1.107e-04 | 1.133e-04 |
| fastDTWF condensed | 1.106e-04 | 1.142e-04 |
| fastDTWF refresh every 50 | 1.107e-04 | 1.134e-04 |

All reported final probabilities are positive. Across the fixed-size sweep,
mass differs from one by at most about `6e-11` (previous FFT). The new backend
and tighter reference agree extremely closely, and the independent previous
backend agrees to about `6e-11` in L1. The fastDTWF modes have relative
segregating-state L1 discrepancies of about `8e-5` after 1,000 generations.

### Single-generation doubling

The source has 28,948 gene copies and the destination 57,896. All fastDTWF modes
use ordinary propagation for this one-generation case; their timing differences
are run-to-run variation, not different algorithms.

| Implementation | Neutral epoch total | Selected epoch total | Selected L1 |
| --- | ---: | ---: | ---: |
| Chebyshev/NUFFT | 0.655 s | 0.588 s | 8.329e-15 |
| Previous FFT | 4.843 s | 4.875 s | 7.827e-14 |
| fastDTWF ordinary | 0.344 s | 0.439 s | 3.374e-06 |

Chebyshev/NUFFT has a clear setup advantage over the previous FFT for this size
change, but fastDTWF is faster when imports are excluded at the tested settings.
This differs from fresh-process comparisons that charge Python and library
imports to every run. The low-memory advantage measured in the earlier report
remains relevant; this experiment does not remeasure peak memory.

Raw data: [fixed epochs](bernstein_epochs_cpu.json) and
[population-size change](bernstein_size_change_cpu.json).

### Timing repeatability check

Two additional fresh workers per selected condensed mode each ran only the
1,000-generation horizon. Their setup followed the same tiny-epoch warmup, but
unlike the original sweep they did not first run the shorter horizons. The
original observations remain in the table above. These follow-ups measure
variability rather than justify replacing an unfavorable observation.

| Mode | Original sweep | Follow-up 1 | Follow-up 2 | Median of all three |
| --- | ---: | ---: | ---: | ---: |
| condensed | 19.631 s | 11.092 s | 11.398 s | 11.398 s |
| refreshed | 21.821 s | 23.430 s | 22.002 s | 22.002 s |

The selected condensed follow-ups took about 11 seconds, versus 19.6 seconds
in the original sweep. Avoid precise speedup or selection-effect claims from
a single observation. All three selected condensed observations remained faster
than the measured 28.5-second replacement epoch. Raw follow-ups, including the
worker commands and diagnostics:
[bernstein_epoch_repeats_cpu.json](bernstein_epoch_repeats_cpu.json).

## Measurement contract

- Fixed epochs use 28,948 gene copies (14,474 diploid effective individuals),
  neutral selection or symmetric underdominance of strength `1e-4`, no mutation,
  and a uniform initial distribution including the absorbing endpoints.
- Each horizon starts from that same initial distribution. Every generation
  consumes the preceding generation's output; timings are not extrapolations.
- Imports and framework initialization are recorded separately and excluded from
  epoch totals. JAX initializes its CPU client; fastDTWF exercises its integration
  machinery on a small, unrelated 16-copy epoch before measurement.
- JAX plans are prepared once per worker. A single executable uses a dynamic
  generation count and a compiled loop. Plan arrays are dynamic arguments. The
  report separates planning, compilation, a one-step first-dispatch calibration,
  and propagation. Epoch totals conservatively include all four, including the
  extra calibration step. Persistent JAX compilation caching is disabled.
- fastDTWF uses its upstream constant-size integration routine, which includes
  coarse graining and, when enabled, condensed-matrix construction inside the
  epoch timer. Modes are ordinary propagation, condensed without intermediate
  refreshes, and condensed with refreshes every 50 generations. Its routine falls
  back to ordinary propagation for single-generation epochs.
- Each case is timed once, sequentially in isolated processes on the same CPU.
  These are exploratory timings, not replicated medians or confidence intervals.
  The order is new backend, previous backend, tighter reference, ordinary
  fastDTWF, condensed fastDTWF, then refreshed fastDTWF.
- The current backend uses `tail_tol=1e-12` with its default accuracy. fastDTWF
  uses `tv_sd=0.1` and `row_eps=1e-8`. These settings do not imply equal error.
  fastDTWF receives `s_het=s`, `s_hom=0` to match symmetric underdominance.

## Accuracy interpretation

The reference is a tighter Chebyshev/NUFFT calculation (`accuracy=1e-13`,
`tail_tol=1e-15`, `nufft_tol=1e-15`), not an exact human-size transition matrix.
The previous FFT implementation provides an independent numerical cross-check.
Small-population tests compare both square and rectangular transitions, repeated
propagation, and condensed refreshes against independently constructed SciPy
binomial transition matrices.

Raw reports retain total mass, minimum probability, endpoint masses, mean
frequency, maximum absolute discrepancy, full-distribution L1 discrepancy, and
segregating-state L1 discrepancy relative to the reference. Relative segregating
L1 divides by the reference's remaining segregating mass, so absorption cannot
hide a large relative error among surviving variants. Distributions are never
clipped or renormalized by the benchmark.

The uniform initial distribution is useful for exercising the full state space,
but is not an equilibrium frequency spectrum. No mutation input, stationary
solve, differentiation, sampling step, or demographic parameter fitting is timed.

## Reproduction

```sh
uv sync --locked --group fastdtwf
mkdir -p .cache/bernstein-baseline
git archive b13114f src | tar -x -C .cache/bernstein-baseline
uv run --group fastdtwf python benchmarks/compare_epochs.py \
  --baseline-source .cache/bernstein-baseline/src \
  --output benchmarks/bernstein_epochs_cpu.json
uv run --group fastdtwf python benchmarks/compare_epochs.py \
  --baseline-source .cache/bernstein-baseline/src \
  --source-gene-copies 28948 --gene-copies 57896 --generations 1 \
  --output benchmarks/bernstein_size_change_cpu.json
```
