# Validation and benchmarks

Run commands from the repository root. Benchmark CLI sizes (`--sizes`,
`--gene-copies`, and `--source-gene-copies`) count allele-bearing copies and must
be even. The scripts convert them to diploid sizes when constructing backends.
`compare_demography.py` uses diploid sizes in its demographic configuration.

```sh
uv run pytest -p no:capture
uv run ruff check src tests benchmarks
uv run ruff format --check src tests benchmarks
uv run ty check src tests benchmarks
uv run pre-commit run --all-files
uv build
uv run python benchmarks/benchmark.py --device cpu --output benchmarks/bernstein_cpu.json
uv run --group fastdtwf python benchmarks/compare_fastdtwf.py --sizes 64 256 1024
```

The spectral benchmark reports compilation, warm execution, sampled forward and
selection-JVP accuracy, backend storage, XLA temporary storage, mass defects and
small-system solve diagnostics. Use `--source-gene-copies` for rectangular
transitions, and `--grad-atol` with `--angular-sensitivity` to evaluate joint value/derivative planning.
Multiple RHS use `jax.vmap`; this processes independent vectors rather than
claiming a shared batched NUFFT speedup.

Use `benchmarks/compare_backends.py` to compare one-generation FFT, NUFFT, and Jacobi actions.
`compare_startup.py` and `compare_epochs.py` also compare these against fastDTWF.

Compare NUFFT encodings with:

```sh
uv run python benchmarks/compare_encodings.py --diagnostics \
  --output benchmarks/encoding_cpu.json
```

Each size, accuracy, encoding, operation and cold repetition gets a fresh
process with persistent compilation caching disabled. The report separates
imports, setup, compilation, first execution, launch-to-first-result and warm
calls. It records every timing sample, geometry size, and XLA temporary memory
(which excludes native FINUFFT workspace). Processes run sequentially, with
encoding order reversed on alternate trials.

`legacy` uses the current execution machinery with the previous degree planner;
`lobatto` uses the sharper interpolation bound; `filtered` adds tapering and
polynomial endpoint correction. An optional `--variants lobatto filtered-matched`
comparison matches polynomial degree except at exact-fallback boundaries; check
the reported degrees. `--grad-atol` with `--angular-sensitivity` enables derivative-target planning.
These planner overrides exist only inside benchmark construction.

`--diagnostics` checks full rows and probability derivatives at a finite set of
boundary-heavy and seeded interior probabilities after timing. It also compares
1, 10 and 100 generations against dense references for sizes up to 1,024,
using singleton and uniform inputs. Sampled maxima are empirical measurements,
not uniform error certificates; default timing inputs are a sinusoidal observable
and uniform mass, at selection strength 0.1.

The [encoding comparison report](https://github.com/mancusolab/drifft/blob/main/benchmarks/ENCODINGS.md)
records the CPU measurements and numerical tradeoffs for the filtered option,
including matched-degree and larger-degree derivative experiments.

Use `benchmarks/compare_demography.py` to compare FFT, NUFFT, and Jacobi with Sequential,
PolynomialKrylov, and RationalKrylov on the same neutral history. Select the epoch
backends with `--backends` and the methods with `--methods`. Each pair runs in a
fresh CPU worker with shared initial mass, singleton mutation input, and FFT size
changes. The report gives cold and warm times, failed interval counts, and final
relative L1 differences against sequential FFT.
The [CPU results](https://github.com/mancusolab/drifft/blob/main/benchmarks/backend_method_cpu.json)
cover all nine pairs on a 150-generation synthetic history and one-generation
actions at 8,192 and 32,768 gene copies. Its
[report](https://github.com/mancusolab/drifft/blob/main/benchmarks/RATIONAL_KRYLOV.md)
records the revision and workload for each measurement.

See the [archived backend comparison](https://github.com/mancusolab/drifft/blob/main/benchmarks/BERNSTEIN.md) for measured limitations.
The older [CPU report](https://github.com/mancusolab/drifft/blob/main/benchmarks/RESULTS.md) and
[fastDTWF comparison](https://github.com/mancusolab/drifft/blob/main/benchmarks/FASTDTWF.md) describe the previous backend and
record earlier revisions. Interpret those measurements against their recorded
implementations rather than the current code.

## FFT numerical validation

`drifft.FFT` uses separate convolution and interpolation grids in production.
Use the existing backend comparison for performance measurements, and the
production numerical regressions for stricter accuracy checks:

```sh
uv run python benchmarks/compare_backends.py \
  --sizes 8192 32768 --repeats 7 --output fft_backends.json
uv run pytest -p no:capture tests/test_fft_numerics.py -q
```

The benchmark uses default backend settings. The numerical tests explicitly
request `atol=1e-12`, `grad_atol=1e-10` with angular sensitivity one and compare
against a 60-digit binomial reference in `tests/_reference.py`. They retain the
original physical probes at 8,192 and 32,768 gene copies, signed value/gradient
actions, transpose checks, and small-system factorization and conditioning
checks. These sampled errors are regression evidence, not uniform floating-point
certificates. All numerical regressions run with the normal test suite.

The derivations and recorded measurements are in
`notes/fft_architecture_investigation.md` and `notes/fft_architecture_results.json`.
The note identifies the commit and commands needed to reproduce the experiment
in a separate checkout.

## Schiffels–Durbin likelihood calculation

These recorded measurements accompany the [sample-likelihood example](examples/schiffels_durbin.md).
They describe the recorded run and have not been remeasured against the current code.

For the default history, all 53 interval checks passed with 128 Krylov vectors.
Total probability mass differed from one by $4.3\times10^{-12}$, and every
population and sample bin was positive. Compared with the previous FFT plus
512-vector [`PolynomialKrylov`](api/methods.md#drifft.PolynomialKrylov) calculation, the likelihood of $K=5$ differed
relatively by $1.3\times10^{-10}$. The largest relative difference across the
conditional sample probabilities was $8.5\times10^{-8}$. These are comparisons
between approximations, not exact-error bounds.

A local float64 CPU run on macOS arm64 measured:

| Stage | Time |
| --- | ---: |
| Geometry setup | 70.7 s |
| JIT tracing and compilation | 57.3 s |
| First synchronized propagation | 44.4 s |
| Synchronized warm propagation | 40.8 s |

The warm call reused the compiled model and restarted from the same ancestral
distribution; both calls gave identical outputs. Timings exclude the sampling
projection and plotting. BLAS/OpenMP thread limits were one, JAX CPU threading
used its default, and persistent compilation caching was disabled. Peak process
memory for the complete run, including both calls, sampling and plotting, was
about 5.1 GB. These single-run measurements do not establish a speedup over
Sequential FFT.
