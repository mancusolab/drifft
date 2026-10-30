"""Independent dense binomial reference and integration contracts."""

# pattern: Functional Core
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import binom

import drifft

jax.config.update("jax_enable_x64", True)


def test_dense_public_backend_and_rectangular_operations():
    """Expose an independent full matrix through the common transition interface."""
    backend = drifft.Dense(12)
    assert isinstance(backend, drifft.AbstractTransitionBackend)
    q = jnp.array([0.0, 1e-12, 0.2, 0.53, 0.9, 1 - 1e-12, 1.0])
    transition = drifft.Transition.from_probabilities(backend, q)
    matrix = jnp.asarray(binom.pmf(np.arange(13)[None, :], 12, np.asarray(q)[:, None]))
    actual_matrix = jax.jit(jax.vmap(transition.expect, out_axes=1))(jnp.eye(13))
    np.testing.assert_allclose(actual_matrix, matrix, atol=2e-15, rtol=2e-14)
    np.testing.assert_allclose(actual_matrix.sum(axis=1), 1, atol=2e-15, rtol=0)
    endpoints = jnp.array([0, -1])
    np.testing.assert_array_equal(actual_matrix[endpoints], jnp.eye(13)[endpoints])
    mass = jnp.cos(jnp.arange(q.size) * 0.3)
    np.testing.assert_allclose(transition.advance(mass), matrix.T @ mass, atol=2e-15)


def test_dense_endpoint_derivatives_include_orders_above_degree():
    """Ordinary AD must differentiate the polynomial at endpoints without custom rules."""
    backend = drifft.Dense(2)
    q = jnp.array([0.0, 0.21, 1.0])
    values = jnp.array([-0.7, 0.4, 1.9])

    def evaluate(q):
        """Evaluate the quadratic expectation at the supplied probabilities."""
        return backend.expect(q, values)

    derivative = evaluate
    for order in range(1, 4):
        previous = derivative

        def derivative(q, previous=previous):
            """Differentiate once along the all-ones probability direction."""
            return jax.jvp(previous, (q,), (jnp.ones_like(q),))[1]

        actual = jax.jit(derivative)(q)
        if order == 1:
            expected = 2 * (
                (1 - q) * (values[1] - values[0]) + q * (values[2] - values[1])
            )
        elif order == 2:
            expected = jnp.full_like(q, 2 * (values[2] - 2 * values[1] + values[0]))
        else:
            expected = jnp.zeros_like(q)
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)


@pytest.mark.parametrize("gene_copies", [1, 0, -2, True, 4.5])
def test_dense_invalid_gene_copy_counts(gene_copies):
    """Reject invalid static dimensions before tracing."""
    with pytest.raises(ValueError, match="gene_copies"):
        drifft.Dense(gene_copies)


def test_dense_shape_validation_and_empty_probability_rows():
    """Match the backend shape contract, including an empty direct evaluation."""
    backend = drifft.Dense(8)
    for q, values in [(jnp.ones((2, 2)), jnp.ones(9)), (jnp.ones(2), jnp.ones(8))]:
        with pytest.raises(ValueError, match="shape|vector|values"):
            jax.jit(backend.expect)(q, values)
    assert backend.expect(jnp.zeros(0), jnp.ones(9)).shape == (0,)


@pytest.mark.parametrize("segregating", [False, True])
def test_dense_demography_with_mutation_selection_and_immigration(segregating):
    """Support changing sizes, matched expectations, and shared frequency parameters."""
    history = drifft.Demography([5, 3, 4], [3, 2])
    rule = drifft.Composition(
        drifft.Mutation(0.003, 0.006), drifft.AdditiveSelection(0.02)
    )
    dense = drifft.DemographicTransition(
        history, backend=drifft.Dense, frequency_map=rule
    )
    fft = drifft.DemographicTransition(history, backend=drifft.FFT, frequency_map=rule)
    initial = jnp.eye(9 if segregating else 11)[2]
    values = jnp.cos(jnp.arange(7 if segregating else 9) * 0.4)

    def objective(model, strength):
        """Update selection without rebuilding the backend or demographic schedule."""
        updated = eqx.tree_at(lambda m: m.frequency_map.maps[1].s_het, model, strength)
        if segregating:
            return updated.advance_segregating(initial, mutation_rate=1e-5).distribution
        return updated.advance(initial).distribution

    strength = jnp.asarray(0.02)
    actual, tangent = eqx.filter_jit(
        lambda s: jax.jvp(lambda x: objective(dense, x), (s,), (jnp.ones_like(s),))
    )(strength)
    expected, reference_tangent = jax.jvp(
        lambda s: objective(fft, s), (strength,), (jnp.ones_like(strength),)
    )
    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=0)
    np.testing.assert_allclose(tangent, reference_tangent, atol=3e-13, rtol=0)
    if not segregating:
        np.testing.assert_allclose(
            actual @ values, initial @ dense.expect(values), atol=3e-14, rtol=0
        )


def test_dense_expected_visits_matches_independent_linear_solve():
    """Use Dense in the standard occupation solver without a backend-specific path."""
    size = 10
    transition = drifft.Transition(drifft.Dense(size), drifft.Underdominance(0.02))
    q = np.asarray(transition.probabilities)[1:-1]
    matrix = binom.pmf(np.arange(1, size)[None, :], size, q[:, None])
    expected = np.linalg.solve(np.eye(size - 1) - matrix.T, np.eye(size - 1)[0])
    result = transition.expected_visits(rtol=1e-12, atol=1e-13)
    assert result.converged
    np.testing.assert_allclose(result.visits, expected, atol=2e-11, rtol=0)


@pytest.mark.parametrize("method_name", ["PolynomialKrylov", "RationalKrylov"])
def test_dense_krylov_power_and_selection_gradient_match_sequential(method_name):
    """Use the shared operator rules with a dense reference transition."""
    backend = drifft.Dense(8)
    initial = jnp.eye(9)[2]
    values = jnp.cos(jnp.arange(9) * 0.4)
    method = getattr(drifft, method_name)(dimension=16, rtol=1e-9, atol=1e-12)

    def evolve(method, strength):
        """Compute the same finite-time count distribution with either method."""
        return method.advance(
            backend,
            initial,
            frequency_map=drifft.AdditiveSelection(strength),
            generations=9,
        ).distribution

    strength = jnp.asarray(0.02)
    expected = evolve(drifft.Sequential(), strength)
    actual = eqx.filter_jit(lambda s: evolve(method, s))(strength)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0)
    gradient = jax.jit(jax.grad(lambda s: evolve(method, s) @ values))(strength)
    reference = jax.grad(lambda s: evolve(drifft.Sequential(), s) @ values)(strength)
    np.testing.assert_allclose(gradient, reference, atol=2e-10, rtol=0)
