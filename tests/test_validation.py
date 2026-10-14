# Functional Core: invalid configuration contracts.
import jax
import jax.numpy as jnp
import pytest

import drifft as wf

jax.config.update("jax_enable_x64", True)


def test_subnormal_positive_tail_tolerance_has_finite_radius():
    plan = wf.build_transition_plan(16, tail_tol=1e-310)
    assert plan.tail_tol == 1e-310
    assert plan.radius == 16


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
        {"gene_copies": 16, "dtype": jnp.float32},
    ],
)
def test_invalid_plan(kwargs):
    with pytest.raises(ValueError):
        wf.build_transition_plan(**kwargs)


def test_shapes_and_solver_options():
    plan = wf.build_transition_plan(16)
    with pytest.raises(ValueError):
        wf.Transition(plan, wf.Underdominance(0.0)).expect(jnp.ones(16))
    with pytest.raises(ValueError):
        wf.Transition(plan, wf.Underdominance(0.0)).expect_segregating(jnp.ones(16))
    with pytest.raises(ValueError):
        wf.Transition(plan, wf.Underdominance(0.0)).expected_visits(solver="cg")  # ty: ignore[invalid-argument-type] -- unsupported solver
    with pytest.raises(ValueError):
        wf.Transition(plan, wf.Underdominance(0.0)).expected_visits(max_steps=0)


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
    plan = wf.build_transition_plan(16)
    with pytest.raises(ValueError, match="finite|integer"):
        wf.Transition(plan, wf.Underdominance(jnp.asarray(0.0))).expected_visits(
            **options
        )


def test_plan_constructor_rejects_inconsistent_metadata():
    from dataclasses import replace

    plan = wf.build_transition_plan(512)
    with pytest.raises(ValueError, match="nodes"):
        replace(plan, nodes=plan.nodes[:-1])
    with pytest.raises(ValueError, match="degree"):
        replace(plan, degree=0)
