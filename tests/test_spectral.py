"""Independent checks of the replacement's sampling and spectral maps."""

# Functional Core: numerical contracts against small independent references.
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import binom

import drifft

jax.config.update("jax_enable_x64", True)


def test_spectral_plan_degree_and_storage():
    plan = drifft.NUFFT(10000, accuracy=1e-10)
    assert 1 <= plan.degree < 10000
    assert plan.spectral_bound <= plan.spectral_tol
    assert all(a.ndim <= 1 for a in jax.tree.leaves(plan))
    tighter = drifft.NUFFT(10000, accuracy=1e-12)
    derivative = drifft.NUFFT(10000, derivative_accuracy=True)
    assert tighter.degree >= plan.degree
    assert derivative.degree >= plan.degree


@pytest.mark.parametrize("degree", [1, 2, 3, 16])
def test_cosine_coefficients(degree):
    from drifft.backends._spectral import cosine_coefficients

    theta = np.pi * np.arange(degree + 1) / degree
    coefficients = np.random.default_rng(degree).normal(size=degree + 1)
    values = np.cos(theta[:, None] * np.arange(degree + 1)) @ coefficients
    np.testing.assert_allclose(
        jax.jit(cosine_coefficients)(jnp.asarray(values)), coefficients, atol=2e-14
    )


@pytest.mark.parametrize("size", [2, 16, 256, 4096])
def test_binomial_samples(size):
    from drifft.backends._spectral import sample_bernstein

    plan = drifft.NUFFT(size)
    values = jnp.asarray(np.random.default_rng(size).normal(size=size + 1))
    actual = jax.jit(lambda x: sample_bernstein(plan, x))(values)
    exact = (
        binom.pmf(np.arange(size + 1)[None, :], size, np.asarray(plan.nodes)[:, None])
        @ values
    )
    np.testing.assert_allclose(actual, exact, atol=2e-12, rtol=0)
    np.testing.assert_allclose(
        sample_bernstein(plan, jnp.ones(size + 1)), 1, atol=1e-14
    )
    np.testing.assert_array_equal(
        actual[jnp.array([0, plan.degree])], values[jnp.array([0, size])]
    )


def test_endpoint_derivatives_and_nufft_modes():
    from drifft.backends._spectral import evaluate_cosine

    q = jnp.array([0.0, 1e-15, 0.1, 0.5, 0.9, 1 - 1e-15, 1.0])
    # q**2 = 3/8 - T1(1-2q)/2 + T2(1-2q)/8.
    coefficients = jnp.array([0.375, -0.5, 0.125])
    fun = jax.jit(lambda q: evaluate_cosine(coefficients, q, 1e-13))
    np.testing.assert_allclose(fun(q), q**2, atol=2e-13)
    _, tangent = jax.jvp(fun, (q,), (jnp.ones_like(q),))
    np.testing.assert_allclose(tangent, 2 * q, atol=2e-12)
    np.testing.assert_allclose(jax.grad(lambda q: fun(q).sum())(q), 2 * q, atol=2e-12)
    second = jax.jvp(
        lambda q: jax.jvp(fun, (q,), (jnp.ones_like(q),))[1], (q,), (jnp.ones_like(q),)
    )[1]
    np.testing.assert_allclose(second, 2, atol=2e-11)


def test_full_endpoint_probability_derivative():
    plan = drifft.NUFFT(32, derivative_accuracy=True)
    values = jnp.asarray(np.random.default_rng(2).normal(size=33))
    q = jnp.array([0.0, 1e-12, 0.2, 0.5, 1 - 1e-12, 1.0])

    def fun(q):
        return drifft.Transition.from_probabilities(plan, q).expect(values)

    tangent = jax.jit(lambda q: jax.jvp(fun, (q,), (jnp.ones_like(q),))[1])(q)
    exact = (
        32
        * binom.pmf(np.arange(32)[None, :], 31, np.asarray(q)[:, None])
        @ np.diff(values)
    )
    np.testing.assert_allclose(tangent, exact, atol=2e-8, rtol=1e-9)


def test_transpose_endpoint_gradients_match_bernstein_identity():
    plan = drifft.NUFFT(16, derivative_accuracy=True)
    values = jnp.asarray(np.random.default_rng(19).normal(size=17))
    q = jnp.array([0.0, 1e-12, 0.2, 0.7, 1 - 1e-12, 1.0])
    mass = jnp.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])

    def loss(q):
        return drifft.Transition.from_probabilities(plan, q).advance(mass) @ values

    actual = jax.jit(jax.grad(loss))(q)
    exact = mass * (
        16
        * binom.pmf(np.arange(16)[None, :], 15, np.asarray(q)[:, None])
        @ np.diff(values)
    )
    np.testing.assert_allclose(actual, exact, atol=2e-9, rtol=1e-9)


@pytest.mark.parametrize("strength", [0.0, 1e-4, 0.2])
def test_expected_visits_implicit_derivative(strength):
    plan = drifft.NUFFT(16, derivative_accuracy=True)

    def total(s):
        return (
            drifft.Transition(plan, drifft.Underdominance(s))
            .expected_visits(rtol=1e-12, atol=1e-13)
            .visits.sum()
        )

    frequencies = np.arange(1, 16) / 16
    q = (frequencies - strength * frequencies * (1 - frequencies)) / (
        1 - 2 * strength * frequencies * (1 - frequencies)
    )
    dq = (
        frequencies
        * (1 - frequencies)
        * (2 * frequencies - 1)
        / (1 - 2 * strength * frequencies * (1 - frequencies)) ** 2
    )
    matrix = binom.pmf(np.arange(1, 16)[None, :], 16, q[:, None])
    derivative = (
        matrix
        * (np.arange(1, 16)[None, :] - 16 * q[:, None])
        / (q * (1 - q))[:, None]
        * dq[:, None]
    )
    system = np.eye(15) - matrix.T
    visits = np.linalg.solve(system, np.eye(1, 15).ravel())
    reference = np.linalg.solve(system, derivative.T @ visits).sum()
    actual = jax.jit(jax.grad(total))(jnp.asarray(strength))
    np.testing.assert_allclose(actual, reference, atol=2e-7, rtol=1e-7)


def test_nufft_centered_mode_conventions_and_imaginary_residual():
    from jax_finufft import nufft2

    theta = jnp.array([0.0, 0.17, 1.0, 2.3, jnp.pi])
    modes = jnp.zeros(9, dtype=jnp.complex128)
    for frequency in [-3, -1, 0, 1, 3]:
        coefficients = modes.at[frequency + 4].set(1)
        actual = jax.jit(lambda a: nufft2(a, theta, iflag=-1, eps=1e-13))(coefficients)
        np.testing.assert_allclose(actual, jnp.exp(-1j * frequency * theta), atol=3e-13)
    symmetric = modes.at[2].set(0.5).at[6].set(0.5)
    actual = nufft2(symmetric, theta, eps=1e-13)
    np.testing.assert_allclose(actual.real, jnp.cos(2 * theta), atol=3e-13)
    assert float(jnp.max(jnp.abs(actual.imag))) < 3e-13
