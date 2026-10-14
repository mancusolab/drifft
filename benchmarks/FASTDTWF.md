# fastDTWF comparison

Recorded measurements below were collected before the Transition-object API
refactor and retained through commit `9e3e694`. Raw JSON is retained as historical
evidence; the updated scripts use `Transition.advance`
and may produce different timings. Run them again for current performance claims.

The optional comparison uses upstream [fastDTWF 0.0.6](https://github.com/jeffspence/fastDTWF/tree/bd36496a64fda137cda5a2d6a97dd9343c97eb46),
pinned to revision `bd36496a64fda137cda5a2d6a97dd9343c97eb46`. Install and run with uv:

```sh
uv run --group fastdtwf python benchmarks/compare_fastdtwf.py \
  --sizes 64 256 1024 4096 --repeats 3 --output benchmarks/fastdtwf_cpu.json
```

## Matched workloads

Both methods use float64 on CPU, the same number of **haploid gene copies**, no
mutation, and the same underdominance probability vector. We pass that vector
directly to fastDTWF rather than translating selection parameters between APIs.

- **One generation:** drifft's full transposed transition versus fastDTWF's
  `mat_multiply(no_fix=False, sfs=False)`. Starting distributions are a singleton
  at count one, uniform mass, and seeded random nonnegative mass normalized to one.
  Absorbing states remain present. Inputs and outputs have `gene_copies + 1` entries.
- **Expected visits:** drifft's GMRES solve versus fastDTWF's `equilibrium_solve`
  with `sfs=True`, `no_fix=False`, and `injection_rate=1/gene_copies`. fastDTWF
  multiplies the injection rate by the population size, giving a unit singleton
  source. Its interior equilibrium solves the same equation
  `(I - Q.T) visits = e1`, including the initial visit. We remove its two absorbing
  entries before comparison. Solves run only up to `--solve-max-m` (default 256).

fastDTWF computes mass-weighted representative probabilities for coarse groups,
so its approximate propagation depends on the input distribution. It does not
provide the signed, fixed linear operator required by our existing matvec tests.
These timings therefore belong to a separate benchmark from `benchmark.py`.
The equations and conventions above were checked against the pinned
[upstream implementation](https://github.com/jeffspence/fastDTWF/blob/bd36496a64fda137cda5a2d6a97dd9343c97eb46/fastDTWF/fastDTWF.py).

## Timing and accuracy

Plans, coarse groups, and fastDTWF log-binomial coefficients are prepared outside
kernel timings and their setup times are reported separately. drifft's plan is
reused across selections; fastDTWF's groups depend on selection and `tv_sd`.
Input generation and conversion between JAX and PyTorch are outside kernel
measurements. fastDTWF reuses a precomputed probability vector, while drifft
recomputes the selection probabilities inside transition evaluation on each
timed transition call. This asymmetry is conservative for drifft's timings.
Repeated timings use the same input, not successive generations.

First-call times include any compilation triggered by that call; subsequent
calls provide warm median/minimum times. JAX results are synchronized. PyTorch
CPU calls are synchronous and run with autograd disabled. fastDTWF import time,
including its eager Numba compilation, is recorded separately. drifft's first
singleton call per size compiles the propagation kernel; other distributions
and selections reuse it. drifft timings are reused across `tv_sd` rows, not
independent repeated measurements. Setup and warm timings should not be added
across report rows to estimate end-to-end execution.

The default fastDTWF sweep uses `tv_sd=0.1,0.01` and `row_eps=1e-8`; drifft uses
`tail_tol=1e-12`, order 12, and its default block settings. These controls are
**not equivalent accuracy guarantees**. fastDTWF also renormalizes each truncated
binomial row; drifft does not. Compare errors alongside times. `--tv-sds`,
`--row-eps`, and `--tail-tol` allow additional accuracy/runtime sweeps.

For sizes up to `--dense-max-m` (default 1024), an independent SciPy binomial
matrix supplies transition L1 errors and expected-visits residuals/errors. NumPy
is used for that independent reference analysis and CPU interchange with PyTorch;
benchmark input construction and drifft kernels use JAX. Above the dense limit,
the report contains differences between methods, mass, and minimum output values,
but no independent exact-error certificate. fastDTWF returns no convergence status;
its solve results must be assessed using the reported diagnostics.

Backend thread policies are not forced to match. The report records PyTorch's
thread count and relevant environment variables; `--threads` changes only
PyTorch's thread count. Timings are single-process, sequential CPU measurements,
not thread-normalized results. No GPU or peak-memory comparison is claimed.
The default benchmark dependency group is optional; integration tests skip its
numerical cases unless installed. Invalid-argument tests run without it.

## Recorded CPU results

[fastdtwf_cpu.json](fastdtwf_cpu.json) contains the complete sweep: macOS 26.4.1
ARM64, JAX 0.11.2, PyTorch 2.14.1 (4 threads), fastDTWF 0.0.6; three warm repeats.
The table selects random starting distributions with selection strength 0.5.
All times are warm medians, excluding setup and compilation.

| Gene copies | fastDTWF `tv_sd` | drifft ms | fastDTWF ms | drifft exact L1 error | fastDTWF exact L1 error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.1 | 0.067 | 2.126 | 1.5e-14 | 2.5e-11 |
| 256 | 0.1 | 0.287 | 8.311 | 2.1e-14 | 6.7e-11 |
| 1024 | 0.1 | 0.831 | 19.215 | 2.7e-14 | 5.1e-5 |
| 1024 | 0.01 | 0.831 | 33.779 | 2.7e-14 | 3.3e-11 |
| 4096 | 0.1 | 2.173 | 51.927 | not measured | not measured |
| 4096 | 0.01 | 2.173 | 137.926 | not measured | not measured |

At 256 copies and selection 0.5, expected visits took 4.854 ms for drifft and
56.954 ms for fastDTWF (`tv_sd=0.1`). Maximum errors against the dense solution
were 8.9e-16 and 1.7e-14, respectively. At 64 and 256 copies these `tv_sd` settings
do not merge states; the report records group counts to make that visible.

These results support faster warm execution for drifft on these particular CPU
workloads. They do not establish a general speedup for fastDTWF's demographic
likelihood, sampling, mutation, or large-population equilibrium workflows. Three
repeats are a preliminary measurement, not an uncertainty estimate; use more
repeats and fresh sweeps for performance claims on other machines.

## Human population size immediately before recent expansion

Measured at constant diploid effective size **14,474**, hence **28,948 gene
copies**. This is the pre-expansion African population size in the
[Tennessen-derived stdpopsim model](https://popsim-consortium.github.io/stdpopsim-docs/stable/catalog.html#sec_catalog_homsap_models_africa_1t12),
not a universal effective size for all modern humans. Selection uses this
package's symmetric underdominance model; these runs are not a fitted model of
human purifying selection.

```sh
uv run --group fastdtwf python benchmarks/compare_fastdtwf.py \
  --sizes 28948 --selections 0 0.00001 0.0001 --tv-sds 0.1 0.01 \
  --repeats 10 --dense-max-m 0 --solve-max-m 0 \
  --output benchmarks/fastdtwf_pre_expansion_cpu.json
```

[Raw results](fastdtwf_pre_expansion_cpu.json) record the same CPU/backend versions
as the preceding sweep and ten warm repetitions. The table selects the seeded
random starting distribution; singleton and uniform distributions are also in
the report and gave similar timings.

| Selection strength | fastDTWF `tv_sd` | drifft ms | fastDTWF ms | Warm speedup | Between-method L1 difference |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1 | 12.59 | 206.94 | 16.4× | 1.94e-05 |
| 0 | 0.01 | 12.59 | 1316.29 | 104.5× | 2.18e-11 |
| 1e-05 | 0.1 | 13.03 | 207.39 | 15.9× | 1.94e-05 |
| 1e-05 | 0.01 | 13.03 | 1430.59 | 109.8× | 2.18e-11 |
| 0.0001 | 0.1 | 13.10 | 210.87 | 16.1× | 1.94e-05 |
| 0.0001 | 0.01 | 13.10 | 1336.82 | 102.0× | 2.19e-11 |

At `tv_sd=0.1`, fastDTWF groups the 28,949 states into 4,654 groups; at `0.01`,
it retains all 28,949 states. The larger speedup at the tighter setting therefore
compares against a configuration with no coarse-group compression. These settings
are not matched error tolerances.

The drifft plan occupied **18.37 MB** and took **2.65 seconds** to construct.
Its first propagation call took **0.48 seconds**, including compilation; later
selections reused the compiled kernel. fastDTWF's group/coefficient setup took
about **1–22 ms**, with import/Numba compilation recorded separately. These costs
mean the warm timing advantage is most useful when the plan is reused.

The reported L1 differences are **between methods, not errors against an exact
reference**. Full dense validation was disabled at this size. Singleton outputs
agreed to approximately 1.03e-11 in L1 at both fastDTWF settings, and all outputs
had total mass within 6e-14 of one. Neither agreement nor mass conservation proves
full-distribution accuracy. This experiment measures one generation at constant
size, with no mutation; it does not benchmark expected-visits solves or a
multi-generation growth trajectory.

## Cold startup and warm reuse at the pre-expansion size

[Isolated-process results](fastdtwf_pre_expansion_startup_cpu.json) measure
28,948 gene copies, uniform input, and selection strengths 0 and 1e-4. Each
configuration has three fresh Python processes, with five synchronized warm
calls after the first result in each process. CPU and dependency versions remain
recorded in the JSON. Run with:

```sh
uv run --group fastdtwf python benchmarks/compare_startup.py \
  --output benchmarks/fastdtwf_pre_expansion_startup_cpu.json
```

The parent imports only the standard library. Each worker imports only its own
backend: drifft/JAX or fastDTWF/PyTorch. Cold time is measured by the parent from
process launch until a flushed marker signals the first completed result. It
includes interpreter startup, imports, input preparation, planning/grouping,
backend initialization, and first-call compilation/execution. Diagnostics and
version lookup happen afterward. fastDTWF's eager Numba compilation is included
in its import phase. JAX persistent compilation caching is disabled. Backend
order reverses on alternating trials. Workers run sequentially.

"Cold" means a fresh process, not a reboot or an empty operating-system file
cache. Installation, uv environment resolution, process shutdown, and the
parent's own startup are excluded. Thread policies remain backend defaults,
with environment settings and PyTorch thread counts recorded. Warm medians are
the median of each process's median; raw timings and cold phase breakdowns are
retained. As in the earlier comparison, fastDTWF caches probabilities while
drifft recomputes them inside the timed call.

| Selection | Backend | Cold first result (s) | Warm call (ms) | Estimated 10 calls (s) | Estimated 100 calls (s) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | drifft | 4.106 | 13.36 | 4.226 | 5.428 |
| 0 | fastDTWF, tv_sd=0.1 | 2.380 | 175.27 | 3.957 | 19.731 |
| 0 | fastDTWF, tv_sd=0.01 | 3.076 | 1081.29 | 12.808 | 110.124 |
| 0.0001 | drifft | 3.772 | 13.56 | 3.894 | 5.115 |
| 0.0001 | fastDTWF, tv_sd=0.1 | 2.386 | 184.48 | 4.046 | 20.650 |
| 0.0001 | fastDTWF, tv_sd=0.01 | 3.283 | 1171.83 | 13.830 | 119.295 |

The 10/100-call columns are **estimates**, computed as median cold time plus
`(calls - 1) * median warm time`; those full sequences were not timed. Repeated
calls use the same uniform input and reuse the same plan/groups. They are not
trajectories that evolve the distribution or change population size.

Against `tv_sd=0.1`, drifft's estimated crossover is **12 calls under neutrality**
and **10 calls at selection 1e-4**. Against `tv_sd=0.01`, it is **two calls** in
both scenarios. These crossover counts are approximate, based on three cold
trials on one machine. The tight fastDTWF setting again leaves all states
ungrouped, and the two packages' approximation settings are not matched accuracy
targets. No new exact-reference accuracy claim is made at this population size.

For example, under neutrality drifft's median import, setup, and first-call phases
were 0.821, 2.540, and 0.540 seconds; fastDTWF at `tv_sd=0.1` took 2.148, 0.002,
and 0.180 seconds. Phase medians need not sum to the median cold total, which also
includes process launch and marker delivery. The measured neutral cold ranges
were 3.952–4.139 seconds for drifft and 2.191–2.742 seconds for fastDTWF at 0.1.

The practical conclusion differs from the warm-only headline: **fastDTWF wins
on a fresh single call; drifft wins with sufficient repeated use** for these
constant-size propagation workloads. Neither conclusion establishes performance
for changing-size demographies or equilibrium solves.
