"""Filtered encoding contracts against binomial and polynomial references."""

# pattern: Functional Core
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import binom

import drifft

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("size,accuracy", [(2, 1e-10), (16, 1e-10), (128, 1e-3)])
def test_filtered_rows_mass_endpoints_and_certificate(size, accuracy):
    """Cover exact fallback and compression, including near-boundary rows."""
    backend = drifft.NUFFT(size, accuracy=accuracy, encoding="filtered")
    q = jnp.asarray(
        np.unique(np.r_[0, 1e-12, 1 / size, np.linspace(0, 1, 41), 1 - 1e-12])
    )
    matrix = jax.jit(jax.vmap(lambda v: backend.expect(q, v)))(jnp.eye(size + 1)).T
    reference = binom.pmf(np.arange(size + 1), size, np.asarray(q)[:, None])
    error = np.max(np.sum(np.abs(matrix - reference), axis=1))
    assert error <= backend.spectral_bound + backend.sampling_bound + 1e-11
    assert backend.spectral_bound <= backend.spectral_tol
    np.testing.assert_allclose(matrix.sum(axis=1), 1, atol=2e-12)
    np.testing.assert_array_equal(matrix[0], np.eye(size + 1)[0])
    np.testing.assert_array_equal(matrix[-1], np.eye(size + 1)[-1])
    if size == 128:
        assert backend.degree < size
        assert backend.taper_width > 0
        assert backend.nodes.size == backend.degree + 2
    else:
        assert backend.degree == size
        assert backend.taper_width == 0


def test_filtered_derivatives_are_of_one_endpoint_corrected_polynomial():
    """Interpolation away from endpoints independently determines endpoint slopes."""
    backend = drifft.NUFFT(128, accuracy=1e-3, encoding="filtered")
    values = jnp.asarray(np.random.default_rng(9).uniform(-1, 1, 129))
    # Interior Chebyshev roots determine the output polynomial without using
    # production sampling, coefficient transforms, or endpoint substitutions.
    x = np.cos(np.pi * (np.arange(backend.degree + 1) + 0.5) / (backend.degree + 1))
    q_fit = jnp.asarray((1 - x) / 2)
    evaluate = jax.jit(lambda q: backend.expect(q, values))
    coefficients = np.polynomial.chebyshev.chebfit(x, evaluate(q_fit), backend.degree)
    q = jnp.array([0.0, 1e-10, 0.13, 0.81, 1 - 1e-10, 1.0])
    for order in (1, 2):
        derivative = evaluate
        for _ in range(order):
            previous = derivative

            def derivative(p, previous=previous):
                """Differentiate the previous polynomial derivative once more."""
                return jax.jvp(previous, (p,), (jnp.ones_like(p),))[1]

        expected = (
            np.polynomial.chebyshev.chebval(
                1 - 2 * q, np.polynomial.chebyshev.chebder(coefficients, m=order)
            )
            * (-2) ** order
        )
        np.testing.assert_allclose(
            jax.jit(derivative)(q), expected, rtol=2e-6, atol=2e-5
        )
    np.testing.assert_allclose(
        np.polynomial.chebyshev.chebval([1, -1], coefficients),
        values[jnp.array([0, 128])],
        atol=2e-11,
    )
    mass = jnp.asarray(np.random.default_rng(10).normal(size=q.size))
    transition = drifft.Transition(backend, probabilities=q)
    np.testing.assert_allclose(
        transition.advance(mass) @ values, mass @ evaluate(q), atol=2e-12
    )
    np.testing.assert_allclose(
        jax.grad(
            lambda p: drifft.Transition(backend, probabilities=p).advance(mass) @ values
        )(q),
        mass * jax.jvp(evaluate, (q,), (jnp.ones_like(q),))[1],
        atol=2e-10,
    )


@pytest.mark.parametrize("encoding", ["bad", "", None, 1])
def test_invalid_encoding(encoding):
    """Reject unknown encoding names at construction."""
    with pytest.raises(ValueError, match="encoding"):
        drifft.NUFFT(128, encoding=encoding)


def test_lobatto_default_and_sharper_planning():
    """Keep the default encoding while reducing the conservative degree."""
    from drifft._backend.nufft import _choose_degree

    backend = drifft.NUFFT(10000)
    assert backend.encoding == "lobatto"
    assert backend.degree == _choose_degree(10000, backend.spectral_tol)
    assert backend.degree < 750  # Previous 4M/d certificate selects 754.
    assert backend.nodes.size == backend.degree + 1
    assert backend.spectral_bound <= backend.spectral_tol


def test_filtered_matches_sampled_fourier_definition():
    """Verify taper weights and alias conventions by an explicit Fourier sum."""
    backend = drifft.NUFFT(128, accuracy=1e-3, encoding="filtered")
    size, modes, width = 128, backend.degree + 1, backend.taper_width
    values = np.random.default_rng(14).uniform(-1, 1, size + 1)
    theta = np.pi * np.arange(2 * modes) / modes
    samples = (
        binom.pmf(np.arange(size + 1), size, np.sin(theta[:, None] / 2) ** 2) @ values
    )
    frequencies = np.arange(1 - modes, modes)
    fourier = (np.exp(-1j * frequencies[:, None] * theta) @ samples) / (2 * modes)
    weights = np.minimum(1, (modes - np.abs(frequencies)) / width)
    q = np.r_[0, 1e-10, np.linspace(0.01, 0.99, 35), 1 - 1e-10, 1]
    angles = 2 * np.arcsin(np.sqrt(q))
    raw = (np.exp(1j * angles[:, None] * frequencies) @ (weights * fourier)).real
    expected = raw + (1 - q) * (values[0] - raw[0]) + q * (values[-1] - raw[-1])
    actual = jax.jit(lambda q, v: backend.expect(q, v))(
        jnp.asarray(q), jnp.asarray(values)
    )
    np.testing.assert_allclose(actual, expected, atol=backend.sampling_bound + 2e-12)
    # Constants and the neutral first moment lie in the flat passband.
    for vector, expected in (
        (np.ones(size + 1), np.ones(q.size)),
        (np.arange(size + 1) / size, q),
    ):
        np.testing.assert_allclose(
            backend.expect(jnp.asarray(q), jnp.asarray(vector)),
            expected,
            atol=backend.sampling_bound + 2e-12,
        )


@pytest.mark.parametrize("size", [2, 3, 4, 16, 100, 10000, 10**9])
@pytest.mark.parametrize("tolerance", [2.5e-14, 1e-7, 0.02])
def test_planners_meet_budget_at_small_and_large_sizes(size, tolerance):
    """Check exact fallback boundaries and compressed rank selection."""
    import math

    from drifft._backend.nufft import (
        _choose_degree,
        _choose_filtered_modes,
        _lobatto_bound,
        _rate,
    )

    degree = _choose_degree(size, tolerance)
    assert 1 <= degree <= size
    assert _lobatto_bound(size, degree) <= tolerance
    modes = _choose_filtered_modes(size, tolerance)
    if modes < size:
        assert modes >= math.ceil(math.sqrt(size))
        assert 1 <= math.ceil(1 / math.atanh(modes / size)) <= modes
        assert 12 * math.exp(-_rate(size, modes)) <= tolerance


@pytest.mark.parametrize("size,tolerance", [(24, 2.5e-5), (54, 2.5e-14)])
def test_lobatto_planner_does_not_skip_sharp_bound_near_exact_degree(size, tolerance):
    """The sharper bound stops before M-1; its last valid degree can suffice."""
    from drifft._backend.nufft import _choose_degree, _lobatto_bound

    expected = next(
        d for d in range(1, size + 1) if _lobatto_bound(size, d) <= tolerance
    )
    assert _choose_degree(size, tolerance) == expected


def test_filtered_derivative_padding_uses_output_degree():
    """Use the same degree convention for both encoding variants."""
    import math

    base = drifft.NUFFT(256, accuracy=1e-6, encoding="filtered")
    padded = drifft.NUFFT(
        256, accuracy=1e-6, encoding="filtered", derivative_accuracy=True
    )
    assert padded.degree == math.ceil(1.5 * base.degree)
