"""Public keyword signatures and result containers."""

# Functional Core: public API and JAX transformation contracts.
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import binom

import drifft

jax.config.update("jax_enable_x64", True)


def test_transition_api_keywords_under_jit():
    plan = drifft.build_transition_plan(gene_copies=512)
    assert isinstance(plan, drifft.TransitionPlan)
    assert plan.gene_copies == 512
    values = jnp.linspace(-1.0, 1.0, 513)
    model = drifft.Transition(plan, selection=drifft.Underdominance(strength=0.1))
    actual = eqx.filter_jit(model.expect)(values=values)
    frequency = np.arange(513) / 512
    probabilities = (frequency - 0.1 * frequency * (1 - frequency)) / (
        1 - 0.2 * frequency * (1 - frequency)
    )
    transition = binom.pmf(np.arange(513)[None, :], 512, probabilities[:, None])
    np.testing.assert_allclose(actual, transition @ values, atol=1e-12)
    segregating = eqx.filter_jit(model.expect_segregating)(values=values[1:-1])
    np.testing.assert_allclose(
        segregating, transition[1:-1, 1:-1] @ values[1:-1], atol=1e-12
    )


def test_expected_visits_result_fields_survive_pytree_roundtrip():
    plan = drifft.build_transition_plan(gene_copies=16)
    result = drifft.Transition(
        plan, drifft.Underdominance(jnp.asarray(0.1))
    ).expected_visits()
    assert isinstance(result, drifft.ExpectedVisitsResult)
    assert bool(result.converged)
    assert result.visits.shape == (15,)
    rebuilt = drifft.ExpectedVisitsResult(
        visits=result.visits,
        status=result.status,
        stats=result.stats,
        residual_norm=result.residual_norm,
        converged=result.converged,
    )
    leaves, structure = jax.tree.flatten(rebuilt)
    restored = jax.tree.unflatten(structure, leaves)
    np.testing.assert_array_equal(restored.visits, result.visits)
