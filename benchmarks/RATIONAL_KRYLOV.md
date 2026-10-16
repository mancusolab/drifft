# Rational Krylov: human pre-expansion evaluation

## Workload

Neutral selection, segregating-site equilibrium, singleton mutation input at rate 1e-8 per gene copy per generation. The supplied private history spans 55,688 generations before recent expansion: 53 intervals, 35 distinct sizes, diploid Ne 3,287–32,060. The initial and final boundaries are 55,940 and 252 generations ago. The demographic input is not included in the repository.

Float64 CPU JAX 0.11.2 on macOS arm64. Fresh worker per backend/geometry version, persistent compilation cache disabled, BLAS/OpenMP thread limits of one with default JAX CPU threading. Workers run sequentially. Timings synchronize outputs and use public per-epoch operations with Python orchestration across changing shapes; they do not measure one monolithic DemographicTransition JIT. FFT handles each interval boundary.

## Original FFT versus Krylov comparison

These measurements preceded geometry compression. Both methods shared the same equilibrium initial vector.

| Measurement | Sequential FFT | RationalKrylov, 128 vectors |
| --- | ---: | ---: |
| Setup | 20.62 s | 55.92 s |
| Cold propagation | 503.26 s | 105.39 s |
| Warm propagation | 472.69 s | 57.18 s |
| First evaluation, including shared initialization | 545.74 s | 183.17 s |
| Peak RSS | 2.90 GB | 4.22 GB |
| Stored geometry | 0.618 GB | 2.673 GB |

Krylov was 2.98× faster for the first evaluation and 8.27× faster warm. Final relative L1 difference from FFT was 5.68e-10; maximum entrywise difference was 2.66e-14. Both cold/warm repeats were identical. All 53 Krylov interval checks passed. The 64-vector configuration took 35.16 s warm but failed 18 interval checks, so it is not used for the accepted comparison. FFT is an approximate reference, not exact truth.

Warm hold-only speedups by interval duration were 1.49× below 500 generations, 8.08× for 500–1,999 generations, and 20.44× for at least 2,000 generations. These groups differ in population size as well as duration.

## Compact indices and quadrature symmetry

The compressed geometry retains one int32 starting index per stencil row and stores only one member of each symmetric Gauss–Jacobi node pair. Even/odd polynomial contributions recover both reflected outputs, including for asymmetric or signed distributions. A central node receives half weight in each contribution. Rank, quadrature order, float64 weights and truncation tolerance are unchanged.

A fresh paired run compares the reconstructed full-node/int64-index implementation with the compact implementation. Both use the same newly computed equilibrium vector, 128 Krylov vectors, and identical solver settings.

| Measurement | Full tables | Compact tables |
| --- | ---: | ---: |
| Stored geometry, including FFT boundaries | 2.673 GB | 1.326 GB |
| Peak process RSS | 3.432 GB | 4.293 GB |
| Setup | 56.26 s | 54.17 s |
| Cold propagation | 108.39 s | 109.59 s |
| Warm propagation | 56.86 s | 58.87 s |

Retained geometry falls by 50.4%, while peak RSS increases by 25.1% in this paired run. Warm runtime changes by +3.5%. The earlier run below showed lower peak RSS, so these measurements do not establish a reliable reduction in peak process memory. RSS includes compilation, temporary arrays and allocator behavior; the cause of the variation has not been isolated.

Final before/after relative L1 difference: **1.89e-12**; maximum entrywise absolute difference: **8.94e-19**. No clipping or normalization is applied. All interval-check failure counts are recorded in the accompanying JSON.

An earlier compact run measured 1.326 GB retained geometry, 3.473 GB peak RSS, 114.33 s cold propagation and 57.51 s warm. Its raw files and the original FFT comparison files were removed by documentation cache cleaning. Those timings were preserved in the execution transcript. The fresh paired results above and aggregate JSON provide the retained comparison.

Each worker executes one cold and one warm pass; these measurements are not confidence intervals. They cover neutral forward evaluation, not selected-model gradients or recent exponential expansion.

A separate compilation-only probe at 28,896 gene copies (1,000 generations, 128 Krylov vectors) found compiled argument storage decreased from 58.32 to 20.39 MB, while compiler-reported temporary storage increased from 80.73 to 90.95 MB. Output storage was unchanged. Peak RSS through compilation decreased from 575.49 to 497.07 MB. This confirms a temporary-storage tradeoff for this shape, but does not explain the full-history RSS reversal; it does not measure execution or accumulation across changing shapes. Raw probe results are retained alongside the paired-worker artifacts.

## Validation and artifacts

- 225 tests passed, including independent full-node/full-binomial decoder references, genuinely truncated stencils, signed/asymmetric inputs, odd/even node counts, JIT, batching, forward and reverse differentiation.
- Ruff, ty, and the documentation build passed.
- Aggregate measurements and source hashes: [geometry_compression_cpu.json](geometry_compression_cpu.json).
- Raw paired-worker logs, final vectors, reconstructed reference and driver are retained locally in `/private/tmp/drifft-geometry-compression/`.

## Run a new comparison

The tracked driver compares the current Sequential FFT and RationalKrylov implementations:

```bash
uv run python benchmarks/compare_demography.py /path/to/history.tsv \
  --cutoff 252 --initialization equilibrium --mutation-rate 1e-8 \
  --krylov-dimension 128 --rtol 1e-8 --atol 1e-12 \
  --equilibrium-rtol 1e-10 --equilibrium-atol 1e-12 \
  --equilibrium-max-steps 4000 --output /tmp/demography-benchmark.json
```

The tab-separated input requires integer `Ne` (diploid population size) and `T_gen` (generations ago) columns, ordered from oldest to youngest with strictly decreasing times. Extra columns are ignored. The cutoff retains boundaries with `T_gen >= cutoff`; it does not interpolate a boundary at the cutoff. At least two boundaries must remain. Each interval applies `duration - 1` generations at its source size, then one FFT step to its destination size. The input is supplied by the user and is not copied into the repository.

The driver uses neutral selection and singleton immigration. `--initialization zero` starts with no segregating mass; the default computes FFT equilibrium once and shares that vector with both workers. `--rank` optionally overrides the Krylov rank. The other flags above expose solver tolerances and limits; FFT geometry uses package defaults, captured by the source hashes.

Each backend runs in a fresh, serial CPU process with float64, persistent compilation caching disabled, and one BLAS/OpenMP thread. JAX CPU threading uses its default. Setup, initialization and propagation are timed separately; imports and process startup are excluded from these timers. One cold and one warm pass restart from the same initial vector. Outputs are synchronized after each hold and boundary. Full-pass timers include Python orchestration and interval diagnostics. This measures public per-epoch calls across changing shapes, not a single compiled `DemographicTransition` call.

The JSON records effective settings, retained history, input and source hashes, Git revision, package versions, devices, peak process RSS, interval diagnostics, final vectors and final FFT/Krylov differences. Output may therefore disclose the supplied demographic history; choose an appropriate destination. A successful converged run exits 0; runtime or convergence failure exits 1, and invalid input exits 2. Nonconvergence still writes measurements with `converged: false`, and those timings should not be treated as an accepted accuracy comparison. Nonfinite output aborts the run. The small synthetic history in `tests/test_demography_benchmark.py` exercises the complete command.

This driver produces new measurements of the current backends. It does not reconstruct the original uncompressed Jacobi implementation or regenerate the historical full-table versus compact-table comparison above.
