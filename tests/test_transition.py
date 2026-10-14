"""Transition object contracts against independent binomial calculations."""

# Functional Core: state ownership, operator orientation, and transformations.
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import binom

import drifft

jax.config.update("jax_enable_x64", True)


def test_selection_model_and_both_transition_directions():
    plan = drifft.build_transition_plan(16)
    transition = drifft.Transition(plan, selection=drifft.Underdominance(strength=0.1))
    x = np.arange(17) / 16
    p = (x - 0.1 * x * (1 - x)) / (1 - 0.2 * x * (1 - x))
    matrix = binom.pmf(np.arange(17)[None, :], 16, p[:, None])
    values = jnp.linspace(-1, 1, 17)
    mass = jnp.arange(1, 18, dtype=jnp.float64) / 153
    np.testing.assert_allclose(
        eqx.filter_jit(transition.expect)(values), matrix @ values, atol=1e-12
    )
    np.testing.assert_allclose(
        eqx.filter_jit(transition.advance)(mass), matrix.T @ mass, atol=1e-12
    )
    np.testing.assert_allclose(
        transition.expect_segregating(values[1:-1]),
        matrix[1:-1, 1:-1] @ values[1:-1],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        transition.advance_segregating(mass[1:-1]),
        matrix[1:-1, 1:-1].T @ mass[1:-1],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        transition.as_lineax().T.mv(mass), matrix.T @ mass, atol=1e-12
    )
    assert float(transition.advance_segregating(mass[1:-1]).sum()) < float(
        mass[1:-1].sum()
    )


def test_explicit_probabilities_preserve_rectangular_transitions():
    plan = drifft.build_transition_plan(8)
    p = jnp.array([0.0, 0.15, 0.7, 1.0])
    transition = drifft.Transition.from_probabilities(plan, p)
    matrix = binom.pmf(np.arange(9)[None, :], 8, np.asarray(p)[:, None])
    values = jnp.arange(9, dtype=jnp.float64)
    mass = jnp.array([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(transition.expect(values), matrix @ values, atol=1e-12)
    np.testing.assert_allclose(transition.advance(mass), matrix.T @ mass, atol=1e-12)
    assert transition.as_lineax().in_structure().shape == (9,)
    assert transition.as_lineax().out_structure().shape == (4,)
    with pytest.raises(ValueError, match="square"):
        transition.expected_visits()


def test_custom_selection_and_rectangular_source_size():
    plan = drifft.build_transition_plan(8)
    transition = drifft.Transition(plan, selection=lambda x: x**2, source_gene_copies=4)
    np.testing.assert_allclose(transition.probabilities, (np.arange(5) / 4) ** 2)
    np.testing.assert_allclose(
        transition.expect(jnp.arange(9, dtype=jnp.float64)),
        8 * (np.arange(5) / 4) ** 2,
        atol=1e-12,
    )


def test_parameters_are_dynamic_and_not_stale_cached_probabilities():
    plan = drifft.build_transition_plan(512)
    transition = drifft.Transition(plan, selection=drifft.Underdominance(0.1))
    traces = []

    @eqx.filter_jit
    def expect(model, values):
        traces.append(1)
        return model.expect(values)

    values = jnp.sin(jnp.arange(513))
    first = expect(transition, values)
    changed = drifft.Transition(plan, selection=drifft.Underdominance(0.2))
    second = expect(changed, values)
    assert len(traces) == 1
    assert not np.allclose(first, second)
    assert changed.plan is transition.plan
    derivative = jax.grad(
        lambda s: drifft.Transition(plan, drifft.Underdominance(s)).expect(values)[200]
    )(0.1)
    h = 1e-5

    def evaluate(s):
        return drifft.Transition(plan, drifft.Underdominance(s)).expect(values)[200]

    np.testing.assert_allclose(
        derivative,
        (evaluate(0.1 + h) - evaluate(0.1 - h)) / (2 * h),
        rtol=1e-4,
        atol=1e-7,
    )


def test_custom_probabilities_drive_expected_visits():
    plan = drifft.build_transition_plan(16)
    p = jnp.arange(17, dtype=jnp.float64) / 16
    transition = drifft.Transition.from_probabilities(plan, p)
    matrix = binom.pmf(np.arange(17)[None, :], 16, np.asarray(p)[:, None])
    rhs = np.zeros(15)
    rhs[0] = 1
    exact = np.linalg.solve(np.eye(15) - matrix[1:-1, 1:-1].T, rhs)
    result = transition.expected_visits()
    assert bool(result.converged)
    np.testing.assert_allclose(result.visits, exact, atol=1e-9)


def test_transition_rejects_ambiguous_or_invalid_configuration():
    plan = drifft.build_transition_plan(16)
    with pytest.raises(ValueError):
        drifft.Transition(plan)
    with pytest.raises(ValueError):
        drifft.Transition.from_probabilities(plan, jnp.ones((2, 2)))
    with pytest.raises(ValueError):
        drifft.Transition.from_probabilities(plan, jnp.array([]))
    with pytest.raises(ValueError):
        drifft.Transition(plan, drifft.Underdominance(0.1), source_gene_copies=0)
    with pytest.raises(ValueError):
        drifft.Underdominance(jnp.array([0.1, 0.2]))


def test_custom_model_gradients_through_advance_and_batched_solves():
    plan = drifft.build_transition_plan(16)
    mass = jnp.zeros(17).at[7].set(1.0)

    def mean_next_count(bias):
        # Construct the callable inside differentiation so its parameter stays
        # connected to the outer trace rather than a precomputed probability map.
        model = drifft.Transition(plan, selection=lambda x: x + bias * x * (1 - x))
        return model.advance(mass) @ jnp.arange(17, dtype=jnp.float64)

    derivative = jax.jit(jax.grad(mean_next_count))(jnp.asarray(0.02))
    np.testing.assert_allclose(derivative, 16 * (7 / 16) * (9 / 16), atol=1e-10)
    results = jax.vmap(
        lambda strength: drifft.Transition(
            plan, drifft.Underdominance(strength)
        ).expected_visits()
    )(jnp.array([0.0, 0.1]))
    assert bool(jnp.all(results.converged))
    assert results.visits.shape == (2, 15)


def test_transition_shape_errors_are_explicit():
    plan = drifft.build_transition_plan(16)
    model = drifft.Transition(plan, drifft.Underdominance(0.0))
    with pytest.raises(ValueError, match="distribution"):
        model.advance(jnp.ones(16))
    with pytest.raises(ValueError, match="interior"):
        model.advance_segregating(jnp.ones(16))
    with pytest.raises(ValueError, match="probability"):
        drifft.Transition(plan, selection=lambda x: jnp.asarray(0.5)).expect(
            jnp.ones(17)
        )
    with pytest.raises(ValueError, match="exactly one"):
        drifft.Transition(
            plan, drifft.Underdominance(0.0), probabilities=jnp.linspace(0, 1, 17)
        )
