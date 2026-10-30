"""Opt-in whole-module CI sharding; default pytest retains the complete suite."""

import pytest

# Keep the shared compiled demographic reference cases in the same process.
# The earlier complete profile assigned about 180s to these modules and 179s
# elsewhere. Assign new modules to shard 2 so additions never escape CI.
_FIRST_SHARD = frozenset(
    {
        "test_demography.py",
        "test_epochs.py",
        "test_gmres_krylov.py",
        "test_krylov_derivatives.py",
        "test_krylov_frechet.py",
        "test_modular_methods.py",
        "test_polynomial_krylov.py",
        "test_power_action.py",
    }
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Expose an explicit CI selector without changing local test defaults."""
    parser.addoption(
        "--ci-shard",
        choices=("1", "2"),
        default=None,
        help="Run one of two complete, disjoint CI groups (default: run all tests).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Select complete modules while reporting ordinary pytest deselections."""
    shard = config.getoption("--ci-shard")
    if shard is None:
        return
    first_shard = {config.rootpath / "tests" / name for name in _FIRST_SHARD}
    selected, deselected = [], []
    for item in items:
        target = (
            selected if (item.path in first_shard) == (shard == "1") else deselected
        )
        target.append(item)
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected
