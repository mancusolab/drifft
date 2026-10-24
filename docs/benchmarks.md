# Validation and benchmarks

Run commands from the repository root.

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
transitions, and `--derivative-accuracy` to evaluate the larger-degree policy.
Multiple RHS use `jax.vmap`; this processes independent vectors rather than
claiming a shared batched NUFFT speedup.

Use `benchmarks/compare_backends.py` to compare FFT and NUFFT.
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
the reported degrees. `--derivative-accuracy` enables the larger-degree heuristic.
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

Use `benchmarks/compare_demography.py` for demographic comparisons, including
Jacobi with RationalKrylov. Historical reports predate the backend/method API
separation and are not performance measurements of the current refactor.

See [historical replacement results](https://github.com/mancusolab/drifft/blob/main/benchmarks/BERNSTEIN.md) for measured limitations.
The older [CPU report](https://github.com/mancusolab/drifft/blob/main/benchmarks/RESULTS.md) and
[fastDTWF comparison](https://github.com/mancusolab/drifft/blob/main/benchmarks/FASTDTWF.md) describe the previous backend and
record earlier revisions; their measurements have not been regenerated for this refactor.
