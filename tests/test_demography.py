"""Demographic timing, dense reference, and transformation contracts."""

# pattern: Functional Core
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.scipy.special import gammaln, xlog1py, xlogy

import drifft

jax.config.update("jax_enable_x64", True)


def dense_transition(source_ne, destination_ne, strength):
    """Build a binomial reference independently of either approximate backend."""
    frequency = jnp.arange(2 * source_ne + 1) / (2 * source_ne)
    # Absorbing rows are exact and independent of selection; differentiate only
    # interior probabilities to avoid log derivatives at zero and one.
    probability = drifft.Underdominance(strength)(frequency[1:-1])[:, None]
    copies = 2 * destination_ne
    count = jnp.arange(copies + 1, dtype=jnp.float64)[None, :]
    interior = jnp.exp(
        gammaln(copies + 1)
        - gammaln(count + 1)
        - gammaln(copies - count + 1)
        + xlogy(count, probability)
        + xlog1py(copies - count, -probability)
    )
    endpoints = jnp.eye(copies + 1)[jnp.array([0, copies])]
    return jnp.concatenate([endpoints[:1], interior, endpoints[1:]])


@pytest.mark.parametrize("backend", [drifft.FFT, drifft.NUFFT])
@pytest.mark.parametrize(
    "sizes,durations", [([8, 5, 11], [3, 2]), ([8, 5], [1]), ([8, 8], [4]), ([8], [])]
)
def test_demographic_evaluation_matches_dense_boundary_history(
    backend, sizes, durations
):
    """Hold the source size and change only on the final interval generation."""
    history = drifft.Demography(ne=sizes, generations=durations)
    model = drifft.DemographicTransition(
        history, backend=backend, selection=drifft.Underdominance(0.07)
    )
    matrix = jnp.eye(2 * sizes[0] + 1)
    for source, destination, duration in zip(
        sizes[:-1], sizes[1:], durations, strict=True
    ):
        matrix = matrix @ jnp.linalg.matrix_power(
            dense_transition(source, source, 0.07), duration - 1
        )
        matrix = matrix @ dense_transition(source, destination, 0.07)
    distribution = jnp.arange(1, 2 * sizes[0] + 2, dtype=jnp.float64)
    distribution /= distribution.sum()
    values = jnp.sin(jnp.arange(2 * sizes[-1] + 1))
    actual = model.advance(distribution)
    expected = model.expect(values)
    assert jnp.allclose(actual, distribution @ matrix, atol=2e-10, rtol=0)
    assert jnp.allclose(expected, matrix @ values, atol=2e-10, rtol=0)
    assert jnp.allclose(actual.sum(), 1, atol=2e-10, rtol=0)
    assert jnp.allclose(actual @ values, distribution @ expected, atol=2e-10, rtol=0)


@pytest.mark.parametrize("backend", [drifft.FFT, drifft.NUFFT])
def test_demographic_selection_gradient_and_batching(backend):
    """Differentiate dynamic selection through size changes and fixed-length loops."""
    model = drifft.DemographicTransition(
        drifft.Demography([8, 5, 11], [3, 2]),
        backend=backend,
        selection=drifft.Underdominance(0.07),
    )
    initial = jnp.eye(17)[3]
    values = jnp.sin(jnp.arange(23))

    def objective(strength):
        """Replace selection while reusing the prepared geometry."""
        updated = eqx.tree_at(lambda m: m.selection.strength, model, strength)
        return updated.advance(initial) @ values

    def reference(strength):
        """Evaluate the same five-generation history using dense binomial matrices."""
        first = dense_transition(8, 8, strength)
        last = dense_transition(5, 5, strength)
        return (
            initial
            @ first
            @ first
            @ dense_transition(8, 5, strength)
            @ last
            @ dense_transition(5, 11, strength)
            @ values
        )

    strength = jnp.asarray(0.07)
    assert jnp.allclose(
        jax.grad(objective)(strength),
        jax.grad(reference)(strength),
        atol=2e-8,
        rtol=2e-6,
    )
    epsilon = 1e-5
    difference = (objective(strength + epsilon) - objective(strength - epsilon)) / (
        2 * epsilon
    )
    assert jnp.allclose(jax.grad(objective)(strength), difference, atol=2e-8, rtol=2e-6)
    batch = jnp.stack([initial, jnp.roll(initial, 1)])
    actual = eqx.filter_jit(eqx.filter_vmap(model.advance))(batch)
    assert jnp.allclose(
        actual, jnp.stack([model.advance(row) for row in batch]), atol=1e-12
    )


def test_demography_normalizes_array_schedule():
    """Copy concrete integer arrays into immutable host metadata."""
    history = drifft.Demography(jnp.array([8, 5, 11]), jnp.array([3, 2]))
    assert history.ne == (8, 5, 11)
    assert history.generations == (3, 2)
    assert history.initial_ne == 8
    assert history.final_ne == 11
    assert history.total_generations == 5


@pytest.mark.parametrize(
    "sizes,durations",
    [
        ([], []),
        ([8, 5], []),
        ([8], [1]),
        ([0, 5], [1]),
        ([8, -1], [1]),
        ([8.5, 5], [1]),
        ([True, 5], [1]),
        ([8, 5], [0]),
        ([8, 5], [-1]),
        ([8, 5], [1.5]),
        ([8, 5], [True]),
        ([[8], [5]], [1]),
    ],
)
def test_demography_rejects_invalid_schedule(sizes, durations):
    """Reject ambiguous lengths, noninteger counts, and nonpositive metadata."""
    with pytest.raises(ValueError):
        drifft.Demography(sizes, durations)


@pytest.mark.parametrize("method,size", [("advance", 10), ("expect", 16)])
def test_demographic_methods_reject_wrong_shape(method, size):
    """Validate source and destination vector lengths at the public boundary."""
    model = drifft.DemographicTransition(
        drifft.Demography([8, 5], [2]),
        backend=drifft.FFT,
        selection=drifft.Underdominance(0.07),
    )
    with pytest.raises(ValueError):
        getattr(model, method)(jnp.ones(size))
