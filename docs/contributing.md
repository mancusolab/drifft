# Development

Install the development tools and Git hooks from the repository root:

```bash
uv sync
uv run pre-commit install
uv run pytest -p no:capture
uv run pre-commit run --all-files
uv build
```

## Test selections and timing

During development, run the affected test files first. For the numerical suite
without benchmark integration or the marked large-population checks, use:

```bash
uv run pytest -q -m "not benchmark and not slow"
```

Default pytest and CI still run the complete suite. Run that suite before
merging, and use per-test timings when investigating runtime:

```bash
uv run pytest -q --durations=30
```

JAX compiles different programs for different static shapes, configurations,
and derivative transformations. Test count alone is therefore a poor runtime
measure. Reuse compiled callables across equivalent parameter cases while
retaining checks of distinct numerical regimes and failure paths.

## Continuous integration

Checks run automatically on pull requests and pushes to main, with manual
dispatch available for other branches. Superseded checks runs are cancelled.
Pre-commit and distribution builds run once; both Python 3.12 and 3.13 run the
complete test suite across two parallel jobs each.

Reproduce an individual test job with:

```bash
uv run pytest -q --ci-shard=1 --durations=20
uv run pytest -q --ci-shard=2 --durations=20
```

The two shards are disjoint and together include every test. Whole modules stay
in one process to reuse JAX compilation, including the shared demographic
reference. New test modules enter shard 2 automatically. Default pytest still
runs everything without a shard option, including benchmark and slow categories.

CI uploads per-shard JUnit timing reports as artifacts. Use those timings to
rebalance the module set in `tests/conftest.py` when costs shift. Parallel jobs
reduce elapsed time by using more concurrent runners; they do not reduce the
numerical coverage or guarantee fewer billed runner minutes. Dependency wheels
and pre-commit environments are cached; tests retain cold JAX compilation.

## Naming and code organization

Use nouns for models, prepared geometry, and results; use verbs for operations.
Keep names aligned with the direction of the transition:

| Name | Meaning |
| --- | --- |
| `backend` | Transition representation (`FFT`, `NUFFT`, `Jacobi`, or reference `Dense`); Jacobi also supplies reduced Krylov geometry |
| `method` | Geometry-free epoch algorithm (`Sequential`, `PolynomialKrylov`, or `RationalKrylov`) |
| `backend_factory`, `boundary_backend_factory` | Prepare transition geometry for demographic holds and size changes |
| `frequency_map` | Selection, mutation, or a composition of both |
| `transition` | Backend bound to selection or explicit source probabilities |
| `probabilities`, `stencil_width` | Computed quantities exposed as properties |
| `expect(values)` | Destination values to source expectations; demographic calls return `DemographicExpectation(values, offset)` |
| `advance(distribution)` | Source mass to destination mass; demographic calls return a result |
| `as_lineax()` | Convert to the linear-operator interface |
| `expected_visits()` | Domain-named solve returning values and convergence diagnostics |
| `prepare_epoch()` | Population inputs to an opaque working problem for a method |

Private helpers use verbs such as `build`, `compute`, `evaluate`, `sample`, and
`solve`. Benchmark entrypoints use `run`; timing functions use `measure`.
Published benchmark JSON retains historical `plan_*` field names so reports
remain comparable; live code calls the prepared object a backend.

`DemographicTransition` composes geometry, frequency maps, and epoch methods.
Do not introduce a public epoch wrapper. `prepare_epoch` returns a PyTree
described by the read-only `EpochProblem` protocol: the working operator,
initial state, exponent, reconstruction, and physical error measurements.
Custom methods use this public contract; projection bookkeeping remains private.
The method computes a power action without knowing the
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
| `_banded.py` | Compact symmetric pentadiagonal operators and reusable Lineax Cholesky solves |
| `_backend/` | Population geometry, sampling, accelerated evaluation and representation-specific assembly |
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

Evolution methods declare `allows_reduced_representation` on the abstract method
contract. True permits reduction when the backend provides it; false requires
population coordinates. Preparation and demographic validation use that declaration rather
than concrete method-type checks. The internal power protocol owns primal/JVP
callbacks; HiJAX calls the sensitivity kernels directly for Krylov pullbacks.
Dense and GMRES resolvent applications share one rational Arnoldi loop, while
Jacobi owns its reduced expectation assembly in `_backend/jacobi.py`.
Use `method.validate(backend, frequency_map)` for public compatibility checks;
Jacobi's `_validate_reduced_map` is an internal reduced-assembly check.

Internal callers import from the owning modules. Public classes are available
directly from `drifft`. Backend implementations live in `_backend`; import
public backend classes from the package root.

Every class, method, and function needs a docstring describing its contract or
purpose. Ruff checks missing docstrings; comments explain numerical identities,
JAX transform boundaries, and memory choices rather than restating operations.

Use raw docstrings (`r"""..."""`) for public numerical APIs. Class docstrings
explain the model and its defining mathematics. Put constructor arguments
and checked errors on `__init__`, never on the class. Omit `**Returns:**`
from `__init__` docstrings. Document generated
Equinox constructors by assigning their `__doc__`, preserving initialization
behavior. Named-tuple constructors use `__new__` instead.
Child methods stay concise: a purpose sentence, `**Arguments:**`, and
`**Returns:**`. Add `**Raises:**` for checked errors. Use `!!! warning`,
`!!! note`, or `!!! tip` admonitions for limitations and practical guidance;
do not repeat the same caveats on every method.
Leave a blank line after each heading and between sections. Properties can use
a short description of the returned value.
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
public Python docstrings with mkdocstrings. On pages with an abstract interface,
place it immediately after the introduction, before concrete implementations,
results, and extension helpers. Use a `??? abstract` block with all public
methods. Document shared method contracts there once; concrete implementations
show `__init__` and only operations with a distinct public contract. Keep
backend-specific numerical guidance in the concrete class description.

Build the site locally with:

```bash
uv run --group docs zensical build --clean --strict
```

Use `--clean` after docstring edits: cached API pages can retain old text even
when an incremental build succeeds. Verify the changed API content in the
generated page.

The generated `site/` directory is ignored by Git. No publishing workflow is
configured. See [validation and benchmarks](benchmarks.md) for numerical checks.

### Audience and scope

Write for statistical and population geneticists familiar with Python. Explain
drifft conventions and JAX/Equinox patterns; assume familiarity with drift,
expectations, derivatives, and conditioning. The tour owns the introductory
walkthrough; examples own distinct biological workflows. Keep API page introductions to one or two sentences, followed by the API
objects. Put precise conventions in the owning docstring; remove duplicates
rather than moving guide preambles above the reference. Short example
admonitions may illustrate individual calls, but complete workflows belong in
the tour or examples. Put algorithm
selection and validation in Numerical guidance; keep derivations in Theory.
Avoid parallel guide pages that repeat the tour or examples.

Use visible section headings for Getting started, Examples, API reference,
Numerical guidance, Theory, and Development. Nest dropdowns only for coherent
subgroups: Algorithms within Theory. Keep API reference pages directly under
the section heading. Keep the page TOC integrated on the left. It includes public methods
but excludes constructor headings (rendered at level 6 by the function template,
below `toc_depth = 5`).
Place prepared-problem helpers in the extension API, not the everyday method
reference. Example-admonition titles are labels, not TOC headings. API member
lists should select constructors and methods; attributes and properties are
omitted by the shared attribute template.

Keep exhaustive input and failure contracts in the API,
derivations in numerical-method chapters, and validation procedures in the
accuracy guide. Link to those pages instead of repeating their guidance.

Treat reader attention as a constraint: lead with the task and recommended
configuration, put required conventions beside the code, and defer optional
choices. Remove introductions that repeat a heading and navigation paragraphs
that merely restate the sidebar. Keep a separate page only when it serves a
distinct task or a substantial reference topic.

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

- `!!! example "Example: Specific task"` gives a short runnable API recipe beside its
  method contract. Place it after the descriptive preamble and before
  `**Arguments:**`, so readers see a concrete call before the detailed contract.
  Keep complete biological analyses on example pages.
- `!!! note "Specific title"` explains interpretation or a modeling convention.
- `!!! tip "Specific title"` gives an optional practical recommendation.
- `!!! warning "Specific title"` identifies a condition that can invalidate a
  calculation or change what its result means, together with the action to take.

Use example blocks selectively when a call pattern needs illustration; do not
add one to every method or repeat argument descriptions inside it. Within a
collapsible abstract-interface block, use expanded `!!!` examples and notes,
never another collapsible block.

Use descriptive sentence-case titles and avoid other nested admonitions. Reserve
collapsible `???` blocks for optional reference or advanced details; assumptions,
required steps, and failure conditions must remain visible. API reference pages
use `??? abstract` for abstract interfaces as described above.

## Support

Report bugs and feature requests in the
[issue tracker](https://github.com/mancusolab/drifft/issues).
