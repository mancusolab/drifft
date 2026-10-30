"""Polynomial normalization and derivative conventions against independent formulas."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import chebyshev
from scipy.special import gegenbauer

from drifft._polynomial import (
    differentiate_cosine_coefficients,
    gegenbauer_moments,
    gegenbauer_values,
)

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("rank", [1, 2, 8])
def test_gegenbauer_values_and_streaming_derivatives(rank):
    """Check basis normalization and differentiated moments without our recurrence."""
    points = np.array([-1.0, -0.7, 0.0, 0.2, 1.0])
    mass = jnp.array([0.3, -0.7, 0.2, 0.5, -0.1])
    coefficients = jnp.sin(jnp.arange(rank) + 0.3)
    polynomials = [
        gegenbauer(n, 1.5) / np.sqrt(3 * (n + 1) * (n + 2) / (2 * (2 * n + 3)))
        for n in range(rank)
    ]
    matrix = np.stack([p(points) for p in polynomials], axis=-1)
    np.testing.assert_allclose(
        jax.jit(lambda z: gegenbauer_values(z, rank))(jnp.asarray(points)),
        matrix,
        atol=2e-12,
        rtol=2e-12,
    )
    for order in (0, 1, 2, rank):
        derivative = np.stack(
            [np.polyder(p, order)(points) for p in polynomials], axis=-1
        )
        actual = jax.jit(
            lambda z, m, order=order: gegenbauer_moments(z, m, rank=rank, order=order)
        )(jnp.asarray(points), mass)
        np.testing.assert_allclose(actual, derivative.T @ mass, atol=2e-10, rtol=2e-12)
        actual = jax.jit(
            lambda z, c, order=order: gegenbauer_moments(
                z, c, rank=rank, order=order, transposed=True
            )
        )(jnp.asarray(points), coefficients)
        np.testing.assert_allclose(
            actual, derivative @ coefficients, atol=2e-10, rtol=2e-12
        )


def test_chebyshev_derivative_coordinate_and_constant_conventions():
    """Differentiate in q rather than z=1-2q, including the constant polynomial."""
    coefficients = jnp.array([0.4, -0.2, 0.7, -0.3, 0.1])
    for _ in range(6):
        expected = chebyshev.chebder(np.asarray(coefficients), scl=-2)
        actual = jax.jit(differentiate_cosine_coefficients)(coefficients)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        coefficients = actual
