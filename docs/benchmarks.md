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

Use `benchmarks/compare_demography.py` for demographic comparisons, including
Jacobi with RationalKrylov. Historical reports predate the backend/method API
separation and are not performance measurements of the current refactor.

See [historical replacement results](https://github.com/mancusolab/drifft/blob/main/benchmarks/BERNSTEIN.md) for measured limitations.
The older [CPU report](https://github.com/mancusolab/drifft/blob/main/benchmarks/RESULTS.md) and
[fastDTWF comparison](https://github.com/mancusolab/drifft/blob/main/benchmarks/FASTDTWF.md) describe the previous backend and
record earlier revisions; their measurements have not been regenerated for this refactor.
