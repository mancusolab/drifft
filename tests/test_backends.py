"""Backend interface, exact small chains, and selection derivative contracts."""

# Functional Core: independent references and transformation checks.
import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import binom

import drifft

jax.config.update("jax_enable_x64", True)


def test_abstract_backend_and_removed_api():
    with pytest.raises(TypeError):
        drifft.AbstractTransitionBackend(32)  # ty: ignore[call-non-callable] -- abstract construction must fail
    assert issubclass(drifft.AbstractTransitionBackend, eqx.Module)
    assert isinstance(drifft.AbstractTransitionBackend, abc.ABCMeta)
    assert not hasattr(drifft, "TransitionPlan")
    assert not hasattr(drifft, "build_transition_plan")


@pytest.mark.parametrize("name", ["FFT", "NUFFT"])
def test_backend_dense_adjoint_and_selection_derivatives(name):
    backend = getattr(drifft, name)(32)
    assert isinstance(backend, drifft.AbstractTransitionBackend)
    mass = jnp.arange(1.0, 26.0)
    mass = mass / mass.sum()
    values = jnp.sin(jnp.arange(33.0))
    strength = jnp.asarray(0.1)

    def model(s):
        return drifft.Transition(
            backend, drifft.Underdominance(s), source_gene_copies=24
        )

    transition = model(strength)
    q = np.asarray(transition.probabilities)
    matrix = binom.pmf(np.arange(33)[None, :], 32, q[:, None])
    np.testing.assert_allclose(
        eqx.filter_jit(transition.expect)(values), matrix @ values, atol=1e-11
    )
    np.testing.assert_allclose(
        eqx.filter_jit(transition.advance)(mass), matrix.T @ mass, atol=1e-11
    )
    np.testing.assert_allclose(
        values @ transition.advance(mass), mass @ transition.expect(values), atol=1e-12
    )

    def f(s):
        return values @ model(s).advance(mass)

    _, tangent = jax.jvp(f, (strength,), (jnp.ones(()),))
    np.testing.assert_allclose(jax.jit(jax.grad(f))(strength), tangent, atol=1e-10)
    h = 1e-5
    np.testing.assert_allclose(
        tangent, (f(strength + h) - f(strength - h)) / (2 * h), atol=1e-8
    )
    np.testing.assert_allclose(
        jax.jit(jax.vmap(f))(jnp.array([0.1, 0.2])),
        jnp.stack([f(0.1), f(0.2)]),
        atol=1e-11,
    )


@pytest.mark.parametrize("name", ["FFT", "NUFFT"])
def test_large_backend_and_repeated_selection_gradient(name):
    """Exercise interpolated FFT blocks and differentiation through a short epoch."""
    backend = getattr(drifft, name)(1024)
    values = jnp.sin(jnp.arange(1025.0) / 17)
    rows = jnp.array([0.0, 0.001, 0.1, 0.499, 0.8, 0.999, 1.0])
    actual = eqx.filter_jit(backend.expect)(rows, values)
    reference = (
        binom.pmf(np.arange(1025)[None, :], 1024, np.asarray(rows)[:, None]) @ values
    )
    np.testing.assert_allclose(actual, reference, atol=2e-9, rtol=0)

    backend = getattr(drifft, name)(32)
    mass = jnp.ones(33) / 33
    reward = jnp.sin(jnp.arange(33.0) / 7)

    def objective(strength):
        transition = drifft.Transition(backend, drifft.Underdominance(strength))
        final = jax.lax.fori_loop(0, 4, lambda _, p: transition.advance(p), mass)
        return final @ reward

    def dense_objective(strength):
        x = np.arange(33) / 32
        q = (x - strength * x * (1 - x)) / (1 - 2 * strength * x * (1 - x))
        matrix = binom.pmf(np.arange(33)[None, :], 32, q[:, None])
        return np.asarray(mass) @ np.linalg.matrix_power(matrix, 4) @ reward

    h = 1e-5
    expected = (dense_objective(0.1 + h) - dense_objective(0.1 - h)) / (2 * h)
    np.testing.assert_allclose(jax.jit(jax.grad(objective))(0.1), expected, atol=2e-8)
    np.testing.assert_allclose(
        jax.grad(jax.checkpoint(objective))(0.1), expected, atol=2e-8
    )
