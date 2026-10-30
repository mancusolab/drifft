"""Independent binomial checks of FFT probability and transpose derivatives."""

# pattern: Functional Core
# Numerical derivative contracts.
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import binom

from drifft import Transition, Underdominance
from drifft.backends.fft import FFT

jax.config.update("jax_enable_x64", True)


def test_fft_probability_derivatives_above_binomial_degree_vanish():
    """Reach the constant-degree recurrence at endpoints and in the interior."""
    backend = FFT(2)
    probabilities = jnp.array([0.0, 0.13, 0.5, 0.89, 1.0])
    values = jnp.array([-0.7, 0.4, 1.9])

    def first(q):
        """Differentiate the quadratic binomial expectation once."""
        return jax.jvp(lambda p: backend.expect(p, values), (q,), (jnp.ones_like(q),))[
            1
        ]

    def second(q):
        """Differentiate the first-order polynomial once more."""
        return jax.jvp(first, (q,), (jnp.ones_like(q),))[1]

    actual = jax.jit(second)(probabilities)
    expected = jnp.full_like(probabilities, 2 * (values[2] - 2 * values[1] + values[0]))
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
    third = jax.jit(lambda q: jax.jvp(second, (q,), (jnp.ones_like(q),))[1])(
        probabilities
    )
    np.testing.assert_allclose(third, 0, atol=2e-14, rtol=0)


@pytest.mark.parametrize("size", [16, 1024])
def test_fft_probability_derivatives_and_transpose(size):
    """Match binomial first and second derivatives, including reflection endpoints."""
    backend = FFT(size)
    q = jnp.array([0.0, 1e-12, 0.03, 0.5, 0.97, 1 - 1e-12, 1.0])
    values = jnp.asarray(np.random.default_rng(size).normal(size=size + 1))
    mass = jnp.array([-0.2, 0.3, -0.7, 0.9, 0.4, -0.5, 0.8])

    def expect(p):
        """Evaluate expectations for the differentiated probability or parameter input."""
        return backend.expect(p, values)

    def first(p):
        """Compute the directional probability derivative for all sampled rows."""
        return jax.jvp(expect, (p,), (jnp.ones_like(p),))[1]

    derivative = jax.jit(first)(q)
    second = jax.jit(lambda p: jax.jvp(first, (p,), (jnp.ones_like(p),))[1])(q)
    references = []
    for order in (1, 2):
        factor = size if order == 1 else size * (size - 1)
        references.append(
            factor
            * binom.pmf(
                np.arange(size - order + 1)[None, :],
                size - order,
                np.asarray(q)[:, None],
            )
            @ np.diff(np.asarray(values), n=order)
        )
    first_tol = 2e-12 if size == 16 else 1e-8
    second_tol = 2e-11 if size == 16 else 1e-7
    np.testing.assert_allclose(derivative, references[0], atol=first_tol, rtol=0)
    np.testing.assert_allclose(second, references[1], atol=second_tol, rtol=0)
    if size > 16:
        boundary = np.asarray(q) != 0.5
        np.testing.assert_allclose(
            derivative[boundary], references[0][boundary], atol=1e-8, rtol=0
        )
        np.testing.assert_allclose(
            second[boundary], references[1][boundary], atol=1e-7, rtol=0
        )
        print(
            "FFT midpoint derivative errors:",
            float(abs(derivative[3] - references[0][3])),
            float(abs(second[3] - references[1][3])),
        )

    def propagated(p):
        """Apply the algebraic transpose at the supplied probabilities."""
        return jax.linear_transpose(lambda v: backend.expect(p, v), values)(mass)[0]

    gradient = jax.jit(jax.grad(lambda p: propagated(p) @ values))(q)
    np.testing.assert_allclose(gradient, mass * references[0], atol=first_tol, rtol=0)
    curvature = jax.jit(
        lambda p: jax.jvp(
            jax.grad(lambda x: propagated(x) @ values), (p,), (jnp.ones_like(p),)
        )[1]
    )(q)
    np.testing.assert_allclose(curvature, mass * references[1], atol=second_tol, rtol=0)
    linear = jax.jit(
        lambda p: jax.jvp(
            lambda x: backend.expect(x, jnp.arange(size + 1.0)),
            (p,),
            (jnp.ones_like(p),),
        )[1]
    )(q)
    np.testing.assert_allclose(linear, size, atol=first_tol, rtol=0)


def test_fft_selection_gradients_match_binomial_chain_rule():
    """Check forward and transpose selection derivatives against the chain rule."""
    size = 16
    backend = FFT(size)
    x = np.arange(size + 1) / size
    strength = 0.4
    heterozygosity = x * (1 - x)
    denominator = 1 - 2 * strength * heterozygosity
    q = (x - strength * heterozygosity) / denominator
    dq = heterozygosity * (2 * x - 1) / denominator**2
    d2q = 4 * heterozygosity**2 * (2 * x - 1) / denominator**3
    rng = np.random.default_rng(6)
    values = jnp.asarray(rng.normal(size=size + 1))
    mass = jnp.asarray(rng.normal(size=size + 1))
    first = (
        size
        * binom.pmf(np.arange(size)[None, :], size - 1, q[:, None])
        @ np.diff(values)
    )
    second = (
        size
        * (size - 1)
        * binom.pmf(np.arange(size - 1)[None, :], size - 2, q[:, None])
        @ np.diff(values, n=2)
    )
    expected_first = np.asarray(mass) @ (first * dq)
    expected_second = np.asarray(mass) @ (second * dq**2 + first * d2q)

    def forward(s):
        """Evaluate the forward scalar objective for the chain-rule reference."""
        return mass @ Transition(backend, Underdominance(s)).expect(values)

    def reverse(s):
        """Evaluate the equivalent transpose objective for the chain-rule reference."""
        return Transition(backend, Underdominance(s)).advance(mass) @ values

    for objective in (forward, reverse):
        np.testing.assert_allclose(
            jax.jit(jax.grad(objective))(strength), expected_first, atol=2e-12, rtol=0
        )
        np.testing.assert_allclose(
            jax.jit(jax.grad(jax.grad(objective)))(strength),
            expected_second,
            atol=2e-11,
            rtol=0,
        )
