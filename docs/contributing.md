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
| `backend` | Prepared geometry (`FFT`, `NUFFT`, or reduced `Jacobi`) |
| `method` | Geometry-free epoch algorithm (`Sequential` or `RationalKrylov`) |
| `boundary_backend` | FFT/NUFFT factory for demographic size changes |
| `frequency_map` | Selection, mutation, or a composition of both |
| `transition` | Backend bound to selection or explicit source probabilities |
| `probabilities`, `stencil_width` | Computed quantities exposed as properties |
| `expect(values)` | Destination values to source expectations |
| `advance(distribution)` | Source mass to destination mass; demographic calls return a result |
| `as_lineax()` | Convert to the linear-operator interface |
| `expected_visits()` | Domain-named solve returning values and convergence diagnostics |

Private helpers use verbs such as `build`, `compute`, `evaluate`, `sample`, and
`solve`. Benchmark entrypoints use `run`; timing functions use `measure`.
Published benchmark JSON retains historical `plan_*` field names so reports
remain comparable; live code calls the prepared object a backend.

`DemographicTransition` composes geometry, frequency maps, and epoch methods.
Do not introduce a public epoch wrapper. Internal prepared problems describe
the working-coordinate operator, initial state, reconstruction, and physical
error measurements. The method computes a power action without knowing the
biological observation likelihood. Shifted linear solves use Lineax internally;
their solver selection is not a public configuration option.

Keep population geometry on backends and algorithm controls on methods. Bind
parameter-dependent operators and factors during evaluation so selection remains
differentiable. Do not retain factors or Krylov bases in demographic results.

Every class, method, and function needs a docstring describing its contract or
purpose. Ruff checks missing docstrings; comments explain numerical identities,
JAX transform boundaries, and memory choices rather than restating operations.

Use raw docstrings (`r"""..."""`) for public numerical APIs. Class docstrings
explain the model and its defining mathematics. Put constructor arguments,
returns, and checked errors on `__init__`, never on the class. Document generated
Equinox constructors by assigning their `__doc__`, preserving initialization
behavior. Named-tuple constructors use `__new__` instead.
Child methods stay concise: a purpose sentence, `**Arguments:**`, and
`**Returns:**`. Add `**Raises:**` for checked errors. Use `!!! warning`,
`!!! note`, or `!!! tip` admonitions for limitations and practical guidance;
do not repeat the same caveats on every method.
Leave a blank line after each heading and between sections. Constructor methods
return `None`; properties can use a short description of the returned value.
Private helpers and internal validation hooks may use concise purpose-only
docstrings.

Distinguish checked shape/configuration errors from unchecked numerical
preconditions. For solvers, document returned failure status, `throw` behavior,
and how batched results are checked. Include transformation and device
limitations where they affect the operation's contract. A class-level summary
provides context for the shorter method contracts.

Use a short defining equation in a class docstring when it clarifies the model,
with symbols explained in the adjacent prose. Keep method docstrings focused
on use and contracts. Put derivations and longer mathematical explanations in
the [model guide](guide/model.md) or the relevant numerical guide.

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
public Python docstrings with mkdocstrings. Present abstract interfaces in
`??? abstract` blocks with all public methods. Concrete implementations show
`__init__` and operations with implementation-specific guidance. Avoid repeating
full interface contracts on each implementation.

Build the site locally with:

```bash
uv run --group docs zensical build --strict
```

The generated `site/` directory is ignored by Git. No publishing workflow is
configured. See [validation and benchmarks](benchmarks.md) for numerical checks.

## Support

Report bugs and feature requests in the
[issue tracker](https://github.com/mancusolab/drifft/issues).
