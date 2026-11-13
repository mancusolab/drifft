"""Independent checks of analytic FFT bounds and actual JAX arithmetic."""

import math
from fractions import Fraction as F

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import binom

from drifft import FFT
from drifft._backend._fft_bounds import (
    _constants,
    _energy,
    _range,
    _tail,
)

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("M", [1, 2, 12, 512])
def test_energy_polynomial_against_binomial_score(M):
    """Check the row energy independently by differentiating its score recurrence."""
    q = 0.37
    # If D^n b = b*s_n, s_(n+1) = (j-Mq)*s_n + q(1-q)*s_n'.
    from numpy.polynomial import Polynomial as P

    j = np.arange(M + 1)
    scores = [P([1.0]) for _ in j]
    for n in range(1, 9):
        scores = [
            P([k, -M]) * s + P([0, 1, -1]) * s.deriv()
            for k, s in zip(j, scores, strict=True)
        ]
        actual = sum(binom.pmf(j, M, q) * np.array([s(q) for s in scores]) ** 2)
        expected = np.polynomial.polynomial.polyval(
            q * (1 - q), [float(c) for c in _energy(M, n)]
        )
        # The independent power-basis evaluation loses digits at high orders.
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-12)
        assert len(_energy(M, n)) <= n + 1


@pytest.mark.parametrize("m", range(2, 17, 2))
def test_interpolation_constants_enclose_signed_weights(m):
    """Compare rational envelopes to independent sampled Lagrange polynomials."""
    alpha, leb, dleb, peano = map(float, _constants(m))
    x = np.linspace(m // 2 - 1, m // 2, 301)
    weights, derivatives = [], []
    for k in range(m):
        roots = np.delete(np.arange(m), k)
        p = np.poly1d(roots, r=True) / np.prod(k - roots)
        weights.append(p(x))
        derivatives.append(np.polyder(p)(x))
    w, dw = np.array(weights), np.array(derivatives)
    assert np.max(np.sum(abs(w), axis=0)) <= leb + 1e-7
    assert np.max(np.sum(abs(dw), axis=0)) <= dleb + 1e-7
    remainder = np.sum(abs(dw) * abs(np.arange(m)[:, None] - x) ** m, axis=0)
    assert np.max(remainder) / math.factorial(m) <= peano + 1e-7
    assert alpha == math.comb(m, m // 2) / 4**m


def test_rational_range_captures_interior_extremum():
    """A grid's endpoint values cannot replace a polynomial range enclosure."""
    lower, upper = _range((F(0), F(1), F(-1)), F(0), F(1))
    assert lower <= 0
    assert F(1, 4) <= upper <= F(17, 64)


@pytest.mark.parametrize("q", [0.001, 0.1, 0.5, 0.99])
def test_tail_bounds_enclose_independent_binomial_tails(q):
    """Both directions include outside-support and wrong-side-of-mean cases."""
    for k in [-1, 0, 1, 10, 50, 99, 100, 101]:
        assert binom.sf(k - 1, 100, q) <= float(_tail(100, k, F(q), upper=True)) + 1e-15
        assert binom.cdf(k, 100, q) <= float(_tail(100, k, F(q), upper=False)) + 1e-15


def _ideal_block(backend, i, q, *, start=None):
    """Build exact-node interpolants using SciPy PMFs, bypassing FFT factors."""
    M = 2 * backend.pop_size
    eta, V = float(backend.eta0[i]), float(backend.variance[i])
    r = (np.log(q) - np.log1p(-q) - eta) * V
    if start is None:
        start = math.floor(r) - backend.interp_order // 2 + 1
    nodes = start + np.arange(backend.interp_order)
    p = expit(eta + nodes / V)
    offset = int(backend.offsets[i])
    bucket, row = next(
        (b, int(rows[0]))
        for b in backend.buckets
        if len(rows := np.flatnonzero(np.asarray(b.output_indices)[:, 0] == offset))
    )
    length = int(bucket.conv_indices[row, 0]) + 1
    lo = int(bucket.indices[row, 0])
    rows = binom.pmf(np.arange(M + 1)[None, :], M, p[:, None])
    rows[:, :lo] = 0
    rows[:, lo + length :] = 0
    t = r - start
    weights, derivatives = [], []
    for k in range(backend.interp_order):
        roots = np.delete(np.arange(backend.interp_order), k)
        # Shift the polynomials to the query to avoid large-power cancellation.
        poly = np.poly1d(roots - t, r=True) / np.prod(k - roots)
        weights.append(poly(0))
        derivatives.append(np.polyder(poly)(0))
    return (
        np.array(weights) @ rows,
        np.array(derivatives) @ rows * V / math.sqrt(q * (1 - q)),
    )


@pytest.mark.parametrize("M,c,m", [(128, 8, 8), (512, 128, 12), (512, 16, 16)])
def test_ideal_interpolants_and_switches_are_bounded(M, c, m):
    """Analytic bounds hold even when the FFT factorization is numerically unstable."""
    backend = FFT(M // 2, min_mode_count=c, interp_order=m)
    report = backend.diagnostics()
    boundaries = np.r_[c / M, expit(np.asarray(backend.edges)), 1 - c / M]
    for i, block in enumerate(report.blocks):
        for q in np.linspace(boundaries[i], boundaries[i + 1], 7):
            actual, grad = _ideal_block(backend, i, q)
            expected = binom.pmf(np.arange(M + 1), M, q)
            expected_grad = expected * (np.arange(M + 1) - M * q) / np.sqrt(q * (1 - q))
            assert np.sum(abs(actual - expected)) <= block.value_bound + 2e-13
            assert (
                np.sum(abs(grad - expected_grad))
                <= block.angular_derivative_bound + 2e-11
            )
    for i, switch in enumerate(report.switches):
        q = switch.probability
        if i == 0 or i == len(report.blocks):
            exact = binom.pmf(np.arange(M + 1), M, q)
            if i == 0:
                exact[backend.boundary_width :] = 0
                approx = _ideal_block(backend, 0, q)[0]
            else:
                exact[: M + 1 - backend.boundary_width] = 0
                approx = _ideal_block(backend, i - 1, q)[0]
        else:
            exact = _ideal_block(backend, i - 1, q)[0]
            approx = _ideal_block(backend, i, q)[0]
        assert np.sum(abs(approx - exact)) <= switch.value_jump_bound + 3e-13


@pytest.mark.parametrize("M", [32, 512, 1024])
def test_actual_jax_rows_jvp_vjp_and_forward(M):
    """Check stable defaults against independent full binomial rows and derivatives."""
    backend = FFT(M // 2)
    diagnostics = backend.diagnostics()
    q = jnp.array([0.0, 0.01, 0.27, 0.381, 0.513, 0.74, 0.99, 1.0])
    basis = jnp.eye(M + 1)

    def rows(p):
        return jax.vmap(lambda v: backend.expect(p, v))(basis).T

    actual, derivative = jax.jit(
        lambda p: jax.jvp(rows, (p,), (jnp.sqrt(p * (1 - p)),))
    )(q)
    reference = binom.pmf(np.arange(M + 1), M, np.asarray(q)[:, None])
    lower = binom.pmf(np.arange(M), M - 1, np.asarray(q)[:, None])
    derivative_reference = M * (
        np.pad(lower, ((0, 0), (1, 0))) - np.pad(lower, ((0, 0), (0, 1)))
    )
    derivative_reference *= np.sqrt(np.asarray(q * (1 - q)))[:, None]
    # These allowances are empirical test tolerances, never reported as certificates.
    assert (
        np.max(np.sum(abs(actual - reference), axis=1))
        <= diagnostics.value_bound + 5e-11
    )
    assert (
        np.max(np.sum(abs(derivative - derivative_reference), axis=1))
        <= diagnostics.angular_derivative_bound + 5e-8
    )
    values = jnp.asarray(np.random.default_rng(M).uniform(-1, 1, M + 1))
    mass = jnp.linspace(-1, 1, len(q))
    forward = jax.jit(lambda v: backend.advance(q, v))(mass)
    np.testing.assert_allclose(
        forward, np.asarray(mass) @ reference, atol=5e-11, rtol=0
    )
    gradient = jax.jit(jax.grad(lambda p: jnp.vdot(mass, backend.expect(p, values))))(q)
    np.testing.assert_allclose(
        gradient * jnp.sqrt(q * (1 - q)),
        mass * (derivative_reference @ values),
        atol=5e-8,
        rtol=0,
    )
    assert backend.error_report.components[0].bound == diagnostics.value_bound
    assert backend.error_report.bound is None


def test_stability_is_separate_from_analytic_error():
    """The allowed small-cutoff geometry exposes amplification, not a certificate."""
    unsafe = FFT(64, min_mode_count=8, interp_order=8).diagnostics()
    stable = FFT(256).diagnostics()
    assert unsafe.amplification_estimate > 1e20
    assert unsafe.value_bound < 0.01
    assert stable.amplification_estimate < 1e4


def test_direct_only_exact_support_and_reflection():
    """Full support has zero analytic error; truncated reflection records its jump."""
    exact = FFT(8).diagnostics()
    assert exact.value_bound == exact.angular_derivative_bound == 0
    assert exact.switches[0].value_jump_bound == 0
    truncated = FFT(128).diagnostics()
    assert not truncated.blocks
    assert truncated.direct_value_bound > 0
    assert truncated.switches[0].kind == "reflection"
    assert truncated.switches[0].value_jump_bound >= 2 * truncated.direct_value_bound


@pytest.mark.parametrize("M,c,m", [(128, 8, 8), (512, 128, 12)])
def test_stencil_value_continuity_and_derivative_jumps(M, c, m):
    """Compare both stencil polynomials at the same knot, without finite differences."""
    backend = FFT(M // 2, min_mode_count=c, interp_order=m)
    diagnostics = backend.diagnostics()
    for i, block in enumerate(diagnostics.blocks):
        q = expit(float(backend.eta0[i]))
        left, dleft = _ideal_block(backend, i, q, start=-m // 2)
        right, dright = _ideal_block(backend, i, q, start=1 - m // 2)
        assert np.sum(abs(left - right)) < 3e-13
        assert (
            np.sum(abs(dleft - dright)) <= block.stencil_derivative_jump_bound + 3e-11
        )


@pytest.mark.parametrize("M", [32, 256, 512])
def test_direct_tail_and_angular_derivative_bounds(M):
    """Check omitted full rows and the direct-only reflection against SciPy."""
    backend = FFT(M // 2)
    diagnostics = backend.diagnostics()
    q = np.linspace(0.001, min(128 / M, 0.5), 51)
    j = np.arange(backend.boundary_width, M + 1)
    omitted = binom.pmf(j, M, q[:, None])
    derivative = omitted * (j - M * q[:, None]) / np.sqrt(q * (1 - q))[:, None]
    assert np.max(np.sum(omitted, axis=1)) <= diagnostics.direct_value_bound + 1e-15
    assert (
        np.max(np.sum(abs(derivative), axis=1))
        <= diagnostics.direct_angular_derivative_bound + 1e-14
    )
    if not diagnostics.blocks:
        low = binom.pmf(np.arange(M + 1), M, 0.5)
        low[backend.boundary_width :] = 0
        assert (
            np.sum(abs(low - low[::-1]))
            <= diagnostics.switches[0].value_jump_bound + 1e-15
        )


def test_balancing_preserves_amplification_estimate():
    """Opposite diagonal rescaling cannot repair intrinsic FFT amplification."""
    balanced = FFT(64, min_mode_count=8, interp_order=8).diagnostics()
    unbalanced = FFT(64, min_mode_count=8, interp_order=8, balance=False).diagnostics()
    np.testing.assert_allclose(
        balanced.amplification_estimate, unbalanced.amplification_estimate, rtol=1e-13
    )
