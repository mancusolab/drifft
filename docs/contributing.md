# Development

Install the development tools and Git hooks from the repository root:

```bash
uv sync
uv run pre-commit install
uv run pytest -p no:capture
uv run pre-commit run --all-files
uv build
```

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
