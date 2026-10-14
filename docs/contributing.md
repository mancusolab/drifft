# Development

Install the development tools and Git hooks from the repository root:

```bash
uv sync
uv run pre-commit install
uv run pytest -p no:capture
uv run pre-commit run --all-files
uv build
```

## Naming and code organization

Use nouns for models, prepared geometry, and results; use verbs for operations.
Keep names aligned with the direction of the transition:

| Name | Meaning |
| --- | --- |
| `backend` | Prepared destination geometry (`FFT` or `NUFFT`) |
| `selection` | Frequency-to-probability model |
| `transition` | Backend bound to selection or explicit source probabilities |
| `probabilities`, `stencil_width` | Computed quantities exposed as properties |
| `expect(values)` | Destination values to source expectations |
| `advance(distribution)` | Source mass to destination mass |
| `as_lineax()` | Convert to the linear-operator interface |
| `expected_visits()` | Domain-named solve returning values and convergence diagnostics |

Private helpers use verbs such as `build`, `compute`, `evaluate`, `sample`, and
`solve`. Benchmark entrypoints use `run`; timing functions use `measure`.
Published benchmark JSON retains historical `plan_*` field names so reports
remain comparable; live code calls the prepared object a backend.

Every class, method, and function needs a docstring describing its contract or
purpose. Ruff checks missing docstrings; comments explain numerical identities,
JAX transform boundaries, and memory choices rather than restating operations.

Keep backend application on the backend class. Extract a helper when it owns a
coherent numerical algorithm, is reused, or is required as a JAX callback.
Primitive binding functions preserve custom derivative rules across transforms;
scan/map callbacks define traced loop bodies. Those are useful boundaries even
when they have one caller. Avoid pass-through wrappers and unused alternate
implementations. Keep helpers beside their callers rather than adding utility
modules for individual functions.

## Documentation

The site uses [Zensical](https://zensical.org/docs/). Documentation dependencies
are separate from the runtime and test environment:

```bash
uv sync --group docs
uv run --group docs zensical serve
```

Open the local URL printed by the server. Markdown pages live in `docs/`;
`zensical.toml` controls navigation and theme settings. API pages render the
public Python docstrings with mkdocstrings.

Build the site locally with:

```bash
uv run --group docs zensical build --strict
```

The generated `site/` directory is ignored by Git. No publishing workflow is
configured. See [validation and benchmarks](benchmarks.md) for numerical checks.

## Support

Report bugs and feature requests in the
[issue tracker](https://github.com/mancusolab/drifft/issues).
