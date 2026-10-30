"""Invalid geometry, vector shape, and solver configuration contracts."""

# Functional Core: invalid configuration contracts.
import jax
import jax.numpy as jnp
import pytest

import drifft as wf

jax.config.update("jax_enable_x64", True)


def test_subnormal_positive_tail_tolerance_has_finite_radius():
    """Keep the stencil radius finite when tolerance is below normal float range."""
    backend = wf.NUFFT(16, tail_tol=1e-310)
    assert backend.tail_tol == 1e-310
    assert backend.radius == 16


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gene_copies": 1},
        {"gene_copies": 16.5},
        {"gene_copies": True},
        {"gene_copies": 16, "tail_tol": 0},
        {"gene_copies": 16, "tail_tol": float("nan")},
        {"gene_copies": 16, "accuracy": 0},
        {"gene_copies": 16, "batch_size": 0},
        {"gene_copies": 16, "nufft_tol": 0},
    ],
)
def test_invalid_backend(kwargs):
    """Reject unsupported dimensions, accuracy controls, and precision."""
    with pytest.raises(ValueError):
        wf.NUFFT(**kwargs)


@pytest.mark.parametrize("backend", [wf.FFT, wf.NUFFT, wf.Jacobi, wf.Dense])
def test_backends_require_float64_configuration(backend):
    """Fail before geometry allocation when the required precision is disabled."""
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="jax_enable_x64"):
            backend(8)


def test_shapes_and_solver_options():
    """Reject malformed vectors and unsupported solver choices at the boundary."""
    backend = wf.NUFFT(16)
    with pytest.raises(ValueError):
        wf.Transition(backend, wf.Underdominance(0.0)).expect(jnp.ones(16))
    with pytest.raises(ValueError):
        wf.Transition(backend, wf.Underdominance(0.0)).expect(
            jnp.ones(16), segregating=True
        )
    with pytest.raises(ValueError):
        wf.Transition(backend, wf.Underdominance(0.0)).expected_visits(solver="cg")  # ty: ignore[invalid-argument-type] -- unsupported solver
    with pytest.raises(ValueError):
        wf.Transition(backend, wf.Underdominance(0.0)).expected_visits(max_steps=0)


@pytest.mark.parametrize(
    "options",
    [
        {"rtol": float("inf")},
        {"atol": float("inf")},
        {"max_steps": 1.5},
        {"max_steps": True},
        {"restart": 2.5},
    ],
)
def test_solver_rejects_invalid_static_options(options):
    """Reject nonfinite tolerances and nonintegral iteration controls."""
    backend = wf.NUFFT(16)
    with pytest.raises(ValueError, match="finite|integer"):
        wf.Transition(backend, wf.Underdominance(jnp.asarray(0.0))).expected_visits(
            **options
        )


def test_nufft_rejects_inconsistent_node_geometry():
    """Validate node count independently of dynamic array values."""
    import equinox as eqx

    backend = wf.NUFFT(512)
    broken = eqx.tree_at(lambda b: b.nodes, backend, backend.nodes[:-1])
    with pytest.raises(ValueError, match="nodes"):
        broken.__check_init__()
