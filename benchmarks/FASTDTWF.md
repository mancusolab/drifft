# fastDTWF comparison

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
recomputes the selection probabilities inside `expected_next_values` on each
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
