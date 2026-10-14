# Validation and CPU measurements

Measured 2026-10-13 on macOS 26.4.1 ARM64, JAX CPU device 0, float64.
Versions: JAX/jaxlib 0.11.2, Lineax 0.1.1, NumPy 2.5.4, SciPy 1.18.1.
No GPU was available. Raw measurements, configuration and versions are in
[cpu.json](cpu.json) and [accuracy_sweep.json](accuracy_sweep.json).

## Commands

```sh
uv sync
uv run pytest -p no:capture
uv run ruff check .
uv run ruff format --check .
uv run ty check src
uv build
git diff --check
uv run python benchmarks/benchmark.py --repeats 5 --output benchmarks/cpu.json
uv run python benchmarks/benchmark.py --sizes 1024 10000 --selections 0 .5 \
  --orders 8 12 --tail-tols 1e-8 1e-12 --solve-max-m 0 --repeats 5 \
  --output benchmarks/accuracy_sweep.json
```

Tests use independent SciPy binomial PMFs for exact references. Only small
systems form dense matrices; large tests evaluate sampled rows without an
M-by-M allocation. Every timed invocation synchronizes with `block_until_ready`.
Times below are medians of five warm invocations, with compilation excluded.
Plan construction includes conversion to device arrays and kernel preparation.
These are single-machine measurements, not hardware-independent speed promises.

## Accuracy

Final verification: **51 tests passed in 27.13 seconds**; Ruff lint/format,
`ty check src`, wheel/sdist builds and whitespace checks passed. Independent
code review closed both documentation findings with no unresolved issues.

The default is 12-point interpolation, tail tolerance `1e-12`, cutoff 128,
spacing one local standard deviation and constant diagonal balancing.

- Local identity tests cover M=100, 1000 and 10000 at p0=.1, .5 and .9, signed
  vectors, offset windows, odd/even lengths and the no-wraparound indexing.
- Every planned log-odds node at M=4096 was compared against the exact sum over
  its own input window. Maximum absolute FFT/factorization error was
  **2.10e-14**. Maximum numerical discarded mass was **5.00e-13**.
- Full small-system tests cover M=16, 64, 256 and 1024 and six selection values
  `{0, .001, .05, .5, .95, 1}`. Vectors include zeros, ones, positive and signed
  random inputs, alternating signs and boundary/central basis vectors. The
  normalized absolute-error requirement is `1e-9`.
- Large sampled-row tests at the same six selections found maximum normalized
  errors **9.09e-14**, **9.85e-13** and **9.65e-13** for M=4096, 10000 and
  65536. The normalization is `max(1, ||v||inf)`. This includes row-sum and
  structured-vector tests and is more demanding than the random-only benchmark.
- All interior seams at M=4096 are tested on both sides and at the seam, along
  with the direct/FFT switch. Five reconstructed kernel rows near the switch
  and seams had maximum row L1 error **1.05e-12** and maximum total negative
  coefficient mass **5.51e-15**. These are measurements, not positivity proofs.
- Signed-vector linearity, the adjoint dot-product identity, exact dense
  transpose comparisons, dynamic-scalar compilation reuse, vmap, and an
  interior selection JVP versus finite differences pass.
- Occupation tests include exact dense references at M=16, 64, 256 and 512;
  M=512 exercises FFT interpolation inside the solve. Tests check internal and
  exact residuals, vector error, nonnegativity tolerance, total absorption time,
  and deliberately exhausted iteration limits.

The tail, interpolation and floating-point errors are distinct. The planned-node
test isolates FFT/factorization from interpolation. The sweep below varies
truncation and interpolation independently; a tighter tail tolerance cannot
remove interpolation error.

## Forward runtime and scaling

Neutral selection, default configuration:

| M | Full operator ms | Banded JAX ms | Dense ms | Transient transpose ms | Plan MB | Plan seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.065 | 0.281 | 0.006 | 0.093 | 0.00048 | 0.016 |
| 256 | 0.204 | 0.790 | 0.041 | 0.273 | 0.00148 | 0.015 |
| 1024 | 0.597 | 3.682 | 0.092 | 0.771 | 0.539 | 0.291 |
| 4096 | 1.022 | 23.664 | — | 1.945 | 2.469 | 0.696 |
| 10000 | 1.697 | 80.755 | — | 4.652 | 6.104 | 1.383 |
| 65536 | 8.684 | 1488.539 | — | 27.393 | 37.007 | 4.122 |

M=64 and 256 use the direct fallback throughout, so their full-operator times
are not FFT timings. **Dense matvecs are faster at every small size measured**;
their table times exclude matrix construction. The banded baseline was slower
than the local operator in this CPU sweep, but recomputes log-PMFs every call.
A selection-specific cached band matrix could change that comparison and is
not implemented here. Plan construction dominates if reuse is low.

At M=65536, forward times for s=.5 and 1 were 9.99 and 12.94 ms; transpose
times were 27.78 and 32.02 ms. Strong selection routes more rows to the direct
boundary path. The M=10000, s=1 run had a noisy median of 9.41 ms versus
1.70 ms at s=0; consult raw minima as well as medians before drawing conclusions.

For M=1024/4096/10000/65536, plans have 58/161/275/767 blocks and
1579/6093/13593/76017 grid nodes, grouped into 2/3/3/4 FFT buckets.
Plan storage per state stays around 0.5–0.6 kB over these sizes. Neutral forward
time grows by about 8.5x between M=4096 and 65536 (16x more states); this finite
range is consistent with the intended scaling but is not an asymptotic proof.

Forward compilation took roughly 0.18–0.43 seconds for M=1024–65536.
At M=65536, compiler-estimated temporary storage was 41.6 MB forward and
139.9 MB transposed. Transposition retains boundary PMF residuals, so its memory
is larger despite linear storage scaling for fixed cutoff. Process peak RSS
over the whole sweep was about 1.35 GB, including compiler, executable caches,
references and both solver benchmarks; it is not per-matvec memory.

An initial version captured the full input vector inside the boundary scan.
Its neutral M=65536 transpose took about 67 ms. Gathering only the two short
boundary windows before the scan reduced this to 27 ms without writing a custom
transpose. The measured adjoint still comes from JAX/Lineax automatic transposition.

## Accuracy versus runtime

Ranges over s=0 and .5; errors use the signed random benchmark vector and sampled
rows. Different benchmark sizes have different random inputs.

| M | Order | Tail tolerance | Forward ms | Worst normalized sampled error |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 8 | 1e-8 | 0.477–0.498 | 8.06e-10 |
| 1024 | 8 | 1e-12 | 0.557–0.570 | 2.85e-11 |
| 1024 | 12 | 1e-8 | 0.497–0.556 | 8.06e-10 |
| 1024 | 12 | 1e-12 | 0.629–0.631 | 1.87e-14 |
| 10000 | 8 | 1e-8 | 1.550–1.623 | 2.90e-10 |
| 10000 | 8 | 1e-12 | 1.642–1.882 | 6.37e-12 |
| 10000 | 12 | 1e-8 | 1.795–1.881 | 2.02e-10 |
| 10000 | 12 | 1e-12 | 1.940–1.971 | 2.42e-14 |

Tightening truncation dominates the loose-tolerance cases. With tight tails,
12-point interpolation substantially reduces the remaining error at modest
runtime cost, motivating the conservative default.

## Occupation solves

All 18 solver/size/selection combinations in the benchmark returned successful
Lineax status and passed the explicit residual check. Raw JSON preserves the
Lineax enum string (`lineax._solution.RESULTS<>` denotes success in this version)
alongside the readable `converged` boolean and iteration statistics.

M=1024, default solver tolerances:

| s | Solver | Warm seconds | Lineax num_steps | Internal residual inf | Exact-model residual inf | Max occupation error |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | GMRES | 0.133 | 9 | 8.43e-16 | 1.06e-14 | 6.03e-13 |
| 0 | BiCGStab | 0.052 | 83 | 8.15e-13 | 8.15e-13 | 6.18e-10 |
| .5 | GMRES | 0.035 | 3 | 2.22e-16 | 2.78e-16 | 4.16e-16 |
| .5 | BiCGStab | 0.0053 | 6 | 8.60e-16 | 1.14e-15 | 5.33e-15 |
| 1 | GMRES | 0.037 | 3 | 2.22e-16 | 2.22e-16 | 7.75e-26 |
| 1 | BiCGStab | 0.0034 | 3 | 2.22e-16 | 2.22e-16 | 1.06e-22 |

GMRES counts outer restart steps; its numbers are not directly comparable to
BiCGStab iterations or matvec counts. First calls at neutral M=1024, including
compilation, took 0.650 s (GMRES) and 0.507 s (BiCGStab). Later selections reused
that solve compilation. The neutral GMRES absorption-time sum was
15.214477925694; BiCGStab gave 15.214477341847. Their small residuals do not by
themselves establish accuracy of larger neutral systems.

## Files and remaining limitations

- `src/drifft/_plan.py`: immutable plans, local factorization and FFT buckets.
- `src/drifft/_operator.py`: full/transient/probability kernels and JAX baseline.
- `src/drifft/_solve.py`: Lineax operators and structured occupation diagnostics.
- `tests/`: factorization, full operator, seams, large rows, transforms, solves
  and validation contracts.
- `pyproject.toml`, `uv.lock`: uv-managed package and reproducible environment.
- `README.md`, this report and benchmark JSON: usage, error contracts and evidence.

No numerical test failures remain in the tested float64 CPU configurations.
GPU speed, float32 reliability, global derivative smoothness and large neutral
occupation accuracy remain unverified. Very aggressive configuration can damage
factor conditioning or interpolation accuracy; finite factors alone do not prove
accuracy. There is no preconditioner, no exact-model error certificate for large
solves, and no cached selection-specific banded backend. Runtime input ranges
are documented preconditions rather than callback-validated checks.
