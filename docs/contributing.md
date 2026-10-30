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
| `backend` | Transition representation (`FFT`, `NUFFT`, `Jacobi`, or reference `Dense`); Jacobi also supplies reduced Krylov geometry |
| `method` | Geometry-free epoch algorithm (`Sequential`, `PolynomialKrylov`, or `RationalKrylov`) |
| `boundary_backend` | Transition backend factory for demographic size changes |
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
parameter-dependent operators and factors during evaluation so selection and
mutation remain differentiable. Do not retain factors or Krylov bases in
demographic results.

The private implementation is organized by responsibility:

| Module | Responsibility |
| --- | --- |
| `_polynomial.py` | Normalized Gegenbauer recurrences, streaming moments and their HiJAX rules; Chebyshev coefficient operations |
| `backends/` | Population geometry, sampling, accelerated evaluation and representation-specific assembly |
| `_evolution/preparation.py` | Adapt representations to power problems and route representation validation |
| `_evolution/problem.py` | Working coordinates, reconstruction, error norms and evolution results |
| `_evolution/projection.py` | Arnoldi bases, projected powers and convergence diagnostics |
| `_evolution/sensitivity.py` | Mathematical power sensitivities and operator pullbacks |
| `_evolution/autodiff.py` | HiJAX integration for operator powers |
| `_evolution/methods.py` | Public method configuration and orchestration |
| `_demography.py` | Population-history chronology and boundary transitions |

Projection and sensitivity kernels do not import backends. Preparation owns
the representation boundary; Sequential uses the one-generation transition,
while Krylov methods may use a reduced representation. Keep Dense's reference
calculation independent of the optimized polynomial kernels.

Internal callers import from the owning modules. The former `_epochs`,
`_krylov`, `_power` and `_jacobi` compatibility modules have been removed;
public classes remain available directly from `drifft`.

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

Use `jax.experimental.hijax.HiPrim` for backend kernels that need custom
differentiation. Declare array shapes and static parameters explicitly, provide
the primal operation, JVP, and batching rule, and derive linearization and reverse
mode with `linearize_from_jvp` and `vjp_from_jvp`. Provide a matched transpose when
the kernel is linear in an input. Keep ordinary JAX operations outside these
boundaries; evolution-level rules for Equinox modules remain separate. HiJAX is
experimental, so verify these transformations against the supported JAX version.

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

### Numerical-method chapters

Use the same top-level sections for each algorithm chapter: **Problem and
representation**, **Computation**, **Cost and reuse**, **Accuracy and controls**,
**Differentiation and adjoints**, and **Use in drifft**. Define dimensions before
using cost formulas. Include preparation, application, transforms, and storage
where relevant; distinguish the cost of one stage from the complete operation.

State what each tolerance controls and what the error estimate excludes.
Separate backend approximation from repeated-power approximation, and distinguish
a defined transpose from an input-dependent Krylov action. Keep application
recipes separate from algorithm descriptions. Link to the API for defaults and
the [algorithm guide](guide/backends.md) for choosing a configuration.

### Admonitions

Keep the main argument and required instructions in ordinary prose. Use
admonitions for a short point a reader should notice beside that argument:

- `!!! note "Specific title"` explains interpretation or a modeling convention.
- `!!! tip "Specific title"` gives an optional practical recommendation.
- `!!! warning "Specific title"` identifies a condition that can invalidate a
  calculation or change what its result means, together with the action to take.

Use descriptive sentence-case titles and avoid nested admonitions. Reserve
collapsible `???` blocks for optional reference or advanced details; assumptions,
required steps, and failure conditions must remain visible. API reference pages
use `??? abstract` for abstract interfaces as described above.

## Support

Report bugs and feature requests in the
[issue tracker](https://github.com/mancusolab/drifft/issues).
