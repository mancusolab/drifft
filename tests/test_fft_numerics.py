"""Independent numerical regressions for the production two-grid FFT backend."""

import math
from decimal import Context, Decimal, Inexact, localcontext

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _reference import decimal_binomial_rows
from scipy.special import expit, logit

from drifft import FFT
from drifft._backend import fft as implementation


@pytest.fixture(scope="module", autouse=True)
def _x64():
    """Enable binary64 for these tests and restore the caller's JAX configuration."""
    with jax.enable_x64():
        yield


def _accuracy_queries(M):
    """Retain the original 61 physical probes at the two large regression sizes.

    Include both sides of representative interfaces from the historical study,
    absorbing endpoints, and log-odds interiors. Query generation depends only
    on M, so changes to the production planner cannot move these regression probes.
    """
    cuts = np.array([c / M for c in (16, 32, 64, 128) if 2 * c < M])
    interfaces = [*cuts, *(1 - cuts)]
    for cutoff in (32, 128):
        if M <= 2 * cutoff:
            continue
        centers = [cutoff]
        while centers[-1] < M - cutoff:
            j = centers[-1]
            centers.append(min(M - cutoff, j + max(1, int(np.sqrt(j * (1 - j / M))))))
        eta = logit(np.asarray(centers) / M)
        edges = expit((eta[:-1] + eta[1:]) / 2)
        interfaces.extend(edges[[0, len(edges) // 2, -1]])
    interfaces = np.array(interfaces)
    interior = expit(np.linspace(logit(1 / M), logit(1 - 1 / M), 16))
    return np.unique(
        np.r_[
            0.0,
            0.5,
            1.0,
            np.nextafter(interfaces, 0),
            interfaces,
            np.nextafter(interfaces, 1),
            interior,
        ]
    )


@eqx.filter_jit
def _angular_jvp(backend, q, values):
    """Evaluate production expectations and their angular JVPs under JIT."""
    return jax.jvp(lambda p: backend.expect(p, values), (q,), (jnp.sqrt(q * (1 - q)),))


@eqx.filter_jit
def _advance(backend, q, mass):
    """Apply the production transpose with dynamic geometry and signed mass."""
    return backend.advance(q, mass)


@eqx.filter_jit
def _coefficient_rows(backend, q, values):
    """Extract value and angular-derivative rows by transposing signed actions."""

    def apply(values):
        """Stack the value and angular JVP before differentiating in values."""
        return jnp.stack(_angular_jvp(backend, q, values))

    return jax.jacrev(apply)(values)


@pytest.mark.parametrize("p", [0.001, 0.1, 0.5, 0.9, 0.999])
def test_centered_cumulant_against_decimal(p):
    """Resolve tiny quadratic terms and large shifts without cancellation or overflow."""
    deltas = np.r_[
        -1000.0,
        -40.0,
        -2.0,
        -0.51,
        -1e-2,
        -1e-5,
        -1e-9,
        0.0,
        1e-9,
        1e-5,
        1e-2,
        0.51,
        2.0,
        40.0,
        1000.0,
    ]
    with localcontext(Context(prec=70)):
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
        implementation._centered_cumulant(p, deltas), expected, rtol=1e-14, atol=0
    )


@pytest.mark.parametrize("stride", [1, 4])
@pytest.mark.parametrize("M,j0,lo,hi", [(128, 8, 0, 50), (1000, 137, 40, 240)])
def test_exact_factorization_for_refined_grid(stride, M, j0, lo, hi):
    """Check the DTD identity against independent high-precision binomial rows."""
    rlo, rhi = -2, 2
    h = 0.25 / (j0 * (1 - j0 / M))
    # Host planning must retain binary64 even when JAX's default dtype is float32.
    with jax.enable_x64(False):
        factors = implementation._build_block(
            M,
            j0,
            lo,
            hi,
            rlo,
            rhi,
            balance=True,
            step=h,
            stride=stride,
        )
    h *= stride / factors.stride
    r = factors.stride * np.arange(rlo, rhi + 1)
    q = expit(math.log(j0 / (M - j0)) + h * r)
    expected, _ = decimal_binomial_rows(M, q)
    actual = (
        np.asarray(factors.a)[:, None]
        * np.exp(-h / 2 * (r[:, None] - (np.arange(lo, hi + 1) - j0)) ** 2)
        * np.asarray(factors.b)[None, :]
    )
    assert max(np.sum(abs(actual - expected[:, lo : hi + 1]), axis=1)) < 1e-14
    # Reconstruct the actual zero-padded convolution, including strided gathers.
    values = np.random.default_rng(321).choice([-1.0, 1.0], hi - lo + 1)
    convolution = np.fft.irfft(
        np.fft.rfft(np.asarray(factors.b) * values, factors.nfft)
        * np.asarray(factors.kernel_fft),
        factors.nfft,
    )
    gathered = (
        np.asarray(factors.a)
        * convolution[factors.length - 1 + factors.stride * np.arange(factors.rows)]
    )
    np.testing.assert_allclose(gathered, actual @ values, atol=2e-14, rtol=0)


def test_convolution_refinement_preserves_interpolation_nodes():
    """Final wide tail windows trigger curvature refinement without changing H."""
    M, j0, h = 512, 8, 0.25 / (8 * (1 - 8 / 512))
    factors = implementation._build_block(M, j0, 0, M, -2, 2, step=h)
    assert factors.stride > 1
    refined_h = h / factors.stride
    curvature_lower = 2 / (M // 2 + 1)
    assert refined_h < curvature_lower
    q = expit(math.log(j0 / (M - j0)) + h * np.arange(-2, 3))
    expected, _ = decimal_binomial_rows(M, q)
    r = factors.stride * np.arange(-2, 3)
    actual = (
        np.asarray(factors.a)[:, None]
        * np.exp(-refined_h / 2 * (r[:, None] - (np.arange(M + 1) - j0)) ** 2)
        * np.asarray(factors.b)[None, :]
    )
    assert np.max(np.sum(abs(actual - expected), axis=1)) < 2e-14


def test_log_concavity_envelope():
    """Check the proved Gaussian envelope around the input factor's mode."""
    M, j0, lo, hi, scale = 128, 8, 0, 40, 0.25
    h = scale / (j0 * (1 - j0 / M))
    factors = implementation._build_block(
        M,
        j0,
        lo,
        hi,
        -5,
        8,
        balance=False,
        step=h,
    )
    h /= factors.stride
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
    bounds = []
    # Keep the physical core fixed and rebuild each order-12 stencil halo.
    for inverse_step in (96.0, 192.0):
        lo = math.floor(-0.03 * inverse_step) - 6
        hi = math.ceil(0.04 * inverse_step) + 6
        bounds.append(
            implementation._interpolation_bound(
                512,
                12,
                float(logit(0.25)),
                inverse_step,
                lo,
                hi,
                gradient=True,
            )
        )
    old, new = bounds
    assert new[0] <= old[0] / 2**12
    assert new[1] <= old[1] / 2**11


@pytest.mark.parametrize("M,min_mode_count", [(512, 16), (1000, 32)])
def test_signed_forward_ad_and_transpose(M, min_mode_count):
    """Check actual forward/JVP/VJP/batching behavior with arbitrary signed data."""
    backend = FFT(
        M // 2,
        min_mode_count=min_mode_count,
        atol=1e-12,
        grad_atol=1e-10,
        angular_sensitivity=1,
    )
    cutoff = min_mode_count / M
    q = np.array(
        [
            0.0,
            0.02,
            np.nextafter(cutoff, 1),
            0.07,
            0.37,
            0.53,
            0.93,
            np.nextafter(1 - cutoff, 0),
            0.98,
            1.0,
        ]
    )
    exact, exact_grad = decimal_binomial_rows(M, q)
    rng = np.random.default_rng(812)
    values = rng.choice([-1.0, 1.0], size=(3, M + 1))
    actual, tangent = jax.vmap(_angular_jvp, in_axes=(None, None, 0))(
        backend, jnp.asarray(q), jnp.asarray(values)
    )
    np.testing.assert_allclose(actual, values @ exact.T, rtol=0, atol=1e-12)
    np.testing.assert_allclose(tangent, values @ exact_grad.T, rtol=0, atol=1e-10)
    mass = rng.normal(size=len(q))
    mass /= sum(abs(mass))
    advanced = _advance(backend, jnp.asarray(q), jnp.asarray(mass))
    assert np.sum(abs(advanced - exact.T @ mass)) < 1e-12
    _, pullback = jax.vjp(
        lambda p: backend.expect(p, jnp.asarray(values[0])), jnp.asarray(q)
    )
    reverse = pullback(jnp.asarray(mass))[0] * np.sqrt(q * (1 - q))
    np.testing.assert_allclose(
        reverse, mass * (exact_grad @ values[0]), rtol=0, atol=1e-10
    )


@pytest.mark.slow
@pytest.mark.parametrize("M", [8192, 32768])
def test_production_accuracy_at_research_sizes(M):
    """Exercise unpatched production at the sizes that exposed the old roundoff floor."""
    backend = FFT(M // 2, atol=1e-12, grad_atol=1e-10, angular_sensitivity=1)
    assert backend.grad_atol is not None
    assert backend.min_mode_count == 32 and backend.interp_order == 12
    assert any(
        np.max(np.diff(np.asarray(b.conv_indices), axis=1)) > 1 for b in backend.buckets
    )
    q = _accuracy_queries(M)
    exact, exact_grad = decimal_binomial_rows(M, q)
    chunks = []
    for start in range(0, len(q), 16):
        part = q[start : start + 16]
        padded = np.pad(part, (0, 16 - len(part)), mode="edge")
        chunks.append(
            np.asarray(_coefficient_rows(backend, padded, jnp.zeros(M + 1)))[
                :, : len(part)
            ]
        )
    rows = np.concatenate(chunks, axis=1)
    value_errors = np.sum(abs(rows[0] - exact), axis=1)
    gradient_errors = np.sum(abs(rows[1] - exact_grad), axis=1)
    value_worst, gradient_worst = np.argmax(value_errors), np.argmax(gradient_errors)
    value_error, gradient_error = (
        value_errors[value_worst],
        gradient_errors[gradient_worst],
    )
    # Empirical CPU regressions, not floating-point certificates in error_report.
    assert value_error < backend.atol
    assert gradient_error < backend.grad_atol
    signs = np.stack(
        [
            np.sign(rows[0][value_worst] - exact[value_worst]),
            np.sign(rows[1][gradient_worst] - exact_grad[gradient_worst]),
            np.random.default_rng(M).choice([-1.0, 1.0], M + 1),
        ]
    )
    values, tangents = jax.vmap(_angular_jvp, in_axes=(None, None, 0))(
        backend, q, signs
    )
    np.testing.assert_allclose(values, signs @ exact.T, atol=backend.atol, rtol=0)
    np.testing.assert_allclose(
        tangents, signs @ exact_grad.T, atol=backend.grad_atol, rtol=0
    )
    mass = np.random.default_rng(M + 1).normal(size=len(q))
    mass /= np.sum(abs(mass))
    assert np.sum(abs(_advance(backend, q, mass) - exact.T @ mass)) < backend.atol
    print(
        f"M={M}: production row L1={value_error:.3g}, angular derivative row L1={gradient_error:.3g}"
    )


@pytest.mark.parametrize("M,grad_atol", [(16, 1e-10), (512, 1e-30)])
def test_direct_only_fallback(M, grad_atol):
    """Empty FFT geometry remains valid for small sizes and strict targets."""
    backend = FFT(
        M // 2,
        atol=1e-12,
        grad_atol=grad_atol,
        angular_sensitivity=1,
    )
    assert not backend.buckets
    q = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    values = np.arange(M + 1) / M
    actual = eqx.filter_jit(backend.expect)(jnp.asarray(q), jnp.asarray(values))
    np.testing.assert_allclose(actual, q, rtol=0, atol=2e-12)


def test_reference_is_independent_of_decimal_context():
    """A caller's arithmetic traps must not change the high-precision oracle."""
    expected = decimal_binomial_rows(8, [0.37])
    with localcontext() as context:
        context.prec = 6
        context.traps[Inexact] = True
        actual = decimal_binomial_rows(8, [0.37])
    np.testing.assert_array_equal(actual, expected)


def test_reference_empty_queries_keep_matrix_shape():
    """An empty batch retains the same row/column convention as populated batches."""
    rows, gradients = decimal_binomial_rows(8, [])
    assert rows.shape == gradients.shape == (0, 9)


def test_nonfinite_factors_raise_a_numerical_error():
    """An overflowing output factor is rejected without depending on assertions."""
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="(?i)nonfinite"):
        implementation._build_block(128, 8, 0, 50, -400, 0, step=1 / 30)


@pytest.mark.parametrize("q", [[-0.1], [1.1], [float("nan")], [float("inf")], [[0.3]]])
def test_reference_rejects_invalid_probabilities(q):
    """Invalid probabilities must not silently become absorbing rows."""
    with pytest.raises(ValueError, match="probabilities"):
        decimal_binomial_rows(8, q)
