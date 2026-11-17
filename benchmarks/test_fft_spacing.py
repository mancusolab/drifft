"""Independent numerical checks for the research prototype, outside the API."""

import math
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from investigate_fft_spacing import (
    advance,
    block,
    centered_cumulant,
    gradient,
    grid,
    prepare,
    reference,
)
from scipy.special import expit

from drifft import FFT


@pytest.mark.parametrize("M", [512, 1000])
def test_baseline_reproduces_production(M):
    """Ensure the prototype's scale-one comparison is the existing backend."""
    actual, _, _ = prepare(M, 1.0, 128, None, 1e-10, 1e-8)
    expected = FFT(M // 2, atol=1e-10, grad_atol=1e-8, angular_sensitivity=1.0)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)
    assert actual.error_report == expected.error_report


@pytest.mark.parametrize("p", [0.001, 0.1, 0.5, 0.9, 0.999])
def test_centered_cumulant_against_decimal(p):
    """The new formula resolves quadratic terms that naive subtraction loses."""
    deltas = np.r_[-2.0, -0.51, -1e-2, -1e-5, -1e-9, 0.0, 1e-9, 1e-5, 1e-2, 0.51, 2.0]
    with localcontext() as context:
        context.prec = 70
        pd = Decimal(p)
        expected = np.array(
            [
                float(
                    (1 - pd + pd * Decimal(float(d)).exp()).ln()
                    - pd * Decimal(float(d))
                )
                for d in deltas
            ]
        )
    np.testing.assert_allclose(
        centered_cumulant(p, deltas), expected, rtol=1e-14, atol=0
    )


@pytest.mark.parametrize("scale", [0.5, 0.25])
@pytest.mark.parametrize("M,j0,lo,hi", [(128, 8, 0, 50), (1000, 137, 40, 240)])
def test_exact_factorization_for_refined_grid(scale, M, j0, lo, hi):
    """Check the DTD identity against independent high-precision binomial rows."""
    rlo, rhi = -5, 8
    factors = block(
        M,
        j0,
        lo,
        hi,
        rlo,
        rhi,
        balance=True,
        scale=scale,
        records=[],
        stable_factors=True,
    )
    h = scale / (j0 * (1 - j0 / M))
    r = np.arange(rlo, rhi + 1)
    q = expit(math.log(j0 / (M - j0)) + h * r)
    expected, _ = reference(M, q)
    actual = (
        np.asarray(factors.a)[:, None]
        * np.exp(-h / 2 * (r[:, None] - (np.arange(lo, hi + 1) - j0)) ** 2)
        * np.asarray(factors.b)[None, :]
    )
    assert max(np.sum(abs(actual - expected[:, lo : hi + 1]), axis=1)) < 1e-14


def test_log_concavity_envelope():
    """Check the proved Gaussian envelope around the input factor's mode."""
    M, j0, lo, hi, scale = 128, 8, 0, 40, 0.25
    factors = block(
        M,
        j0,
        lo,
        hi,
        -5,
        8,
        balance=False,
        scale=scale,
        records=[],
        stable_factors=True,
    )
    h = scale / (j0 * (1 - j0 / M))
    j = np.arange(max(lo, 1), min(hi, M - 1) + 1)
    curvature = min(np.log1p(1 / j) + np.log1p(1 / (M - j)))
    mu = curvature - h
    assert mu > 0
    b = np.asarray(factors.b)
    b = b / max(b)
    distance = abs(np.arange(len(b)) - np.argmax(b))
    assert np.all(b <= np.exp(-mu * distance * (distance - 1) / 2) * (1 + 1e-14))
    assert np.sum(b * b) <= 3 + math.sqrt(math.pi / mu)


def test_refinement_bounds():
    """Shrinking a nested halo gives the stated powers of two in both errors."""
    _, old = grid(512, 128, 1.0, 12, 1.0, 1.0)
    _, new = grid(512, 128, 1.0, 12, 1.0, 0.5)
    assert new[0] <= old[0] / 2**12
    assert new[1] <= old[1] / 2**11


@pytest.mark.parametrize("separate_grids", [False, True])
def test_signed_forward_ad_and_transpose(separate_grids):
    """Check actual forward/JVP/VJP/batching behavior with arbitrary signed data."""
    backend, records, _ = prepare(
        512,
        0.25,
        16,
        None,
        1e-12,
        1e-10,
        stable_factors=True,
        separate_grids=separate_grids,
    )
    if separate_grids:
        assert max(r["stride"] for r in records) > 1
        assert max(r["curvature_ratio_upper"] for r in records) < 1
    q = np.array([0.0, 0.02, 0.0312501, 0.07, 0.37, 0.53, 0.93, 0.9687499, 0.98, 1.0])
    exact, exact_grad = reference(512, q)
    rng = np.random.default_rng(812)
    values = rng.choice([-1.0, 1.0], size=(3, 513))
    actual, tangent = jax.vmap(gradient, in_axes=(None, None, 0))(
        backend, jnp.asarray(q), jnp.asarray(values)
    )
    np.testing.assert_allclose(actual, values @ exact.T, rtol=0, atol=1e-12)
    np.testing.assert_allclose(tangent, values @ exact_grad.T, rtol=0, atol=1e-10)
    mass = rng.normal(size=len(q))
    mass /= sum(abs(mass))
    advanced = advance(backend, jnp.asarray(q), jnp.asarray(mass))
    assert np.sum(abs(advanced - exact.T @ mass)) < 1e-12
    _, pullback = jax.vjp(
        lambda p: backend.expect(p, jnp.asarray(values[0])), jnp.asarray(q)
    )
    reverse = pullback(jnp.asarray(mass))[0] * np.sqrt(q * (1 - q))
    np.testing.assert_allclose(
        reverse, mass * (exact_grad @ values[0]), rtol=0, atol=1e-10
    )
