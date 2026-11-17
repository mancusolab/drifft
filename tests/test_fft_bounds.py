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
from drifft._backend.fft import _energy, _range

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
def test_signed_weight_bound(m):
    """Prove the common Lebesgue envelope by exact polynomial range enclosures."""
    # No sampled maximizer: Fraction-valued polynomial coefficients and common
    # rational subintervals enclose the sum of absolute weights everywhere.
    weights = []
    for k in range(m):
        coefficients = np.array([F(1)], dtype=object)
        for j in range(m):
            if j != k:
                coefficients = np.convolve(
                    coefficients,
                    np.array([F(-j, k - j), F(1, k - j)], dtype=object),
                )
        weights.append(tuple(coefficients))
    for cell in range(16):
        lo, hi = F(m // 2 - 1) + F(cell, 16), F(m // 2 - 1) + F(cell + 1, 16)
        bound = sum(max(abs(x) for x in _range(p, lo, hi, depth=0)) for p in weights)
        assert bound <= 2
        derivatives = [tuple(i * p[i] for i in range(1, len(p))) for p in weights]
        grad_bound = sum(
            max(abs(x) for x in _range(p, lo, hi, depth=0)) for p in derivatives
        )
        assert grad_bound <= 4


def test_rational_range_captures_interior_extremum():
    """A grid's endpoint values cannot replace a polynomial range enclosure."""
    lower, upper = _range((F(0), F(1), F(-1)), F(0), F(1))
    assert lower <= 0
    assert F(1, 4) <= upper <= F(17, 64)


def _ideal_block(backend, i, q, *, start=None):
    """Build exact-node interpolants using SciPy PMFs, bypassing FFT factors."""
    M = 2 * backend.pop_size
    eta, V = float(backend.eta0[i]), float(backend.inverse_step[i])
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
    bound = backend.interpolation_bound + backend.sampling_bound
    boundaries = np.r_[c / M, expit(np.asarray(backend.edges)), 1 - c / M]
    for i in range(len(backend.eta0)):
        for q in np.linspace(boundaries[i], boundaries[i + 1], 7):
            actual, _ = _ideal_block(backend, i, q)
            expected = binom.pmf(np.arange(M + 1), M, q)
            assert np.sum(abs(actual - expected)) <= bound + 2e-13
    for i, q in enumerate(boundaries):
        if i == 0 or i == len(backend.eta0):
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
        assert np.sum(abs(approx - exact)) <= 2 * bound + 3e-13


@pytest.mark.parametrize("M", [32, 512, 1024])
@pytest.mark.parametrize("atol", [1e-10, 4e-15])
def test_actual_jax_rows_jvp_vjp_and_forward(M, atol):
    """Check stable defaults against independent full binomial rows and derivatives."""
    backend = FFT(M // 2, atol=atol)
    bound = backend.interpolation_bound + backend.sampling_bound
    q = jnp.array([0.0, 0.01, 0.27, 0.381, 0.513, 0.74, 0.99, 1.0])
    basis = jnp.eye(M + 1)

    def rows(p, basis):
        """Extract coefficient rows by applying the operator to basis vectors."""
        return jax.vmap(lambda v: backend.expect(p, v))(basis).T

    actual, derivative = jax.jit(
        lambda p, basis: jax.jvp(
            lambda x: rows(x, basis), (p,), (jnp.sqrt(p * (1 - p)),)
        )
    )(q, basis)
    reference = binom.pmf(np.arange(M + 1), M, np.asarray(q)[:, None])
    lower = binom.pmf(np.arange(M), M - 1, np.asarray(q)[:, None])
    derivative_reference = M * (
        np.pad(lower, ((0, 0), (1, 0))) - np.pad(lower, ((0, 0), (0, 1)))
    )
    derivative_reference *= np.sqrt(np.asarray(q * (1 - q)))[:, None]
    # These allowances are empirical test tolerances, never reported as certificates.
    assert np.max(np.sum(abs(actual - reference), axis=1)) <= bound + 5e-11
    assert np.max(np.sum(abs(derivative - derivative_reference), axis=1)) <= 5e-8
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
    assert backend.error_report.components[0].bound == backend.interpolation_bound
    assert backend.error_report.bound is None


@pytest.mark.parametrize("M", [32, 512, 1024])
@pytest.mark.parametrize("atol", [1e-6, 1e-10, 4e-15])
def test_atol_plans_interpolation_and_sampling(M, atol):
    """Automatic preparation meets both analytic allocations, including the floor."""
    backend = FFT(M // 2, atol=atol)
    assert backend.interpolation_bound <= atol / 4
    assert backend.sampling_bound <= atol / 4
    assert [c.name for c in backend.error_report.components] == [
        "interpolation",
        "sampling",
        "roundoff",
    ]
    assert backend.error_report.bound is None


def test_explicit_order_is_an_override():
    """A manual low order is reported honestly, as with Jacobi's manual rank."""
    backend = FFT(256, interp_order=2, min_mode_count=8)
    assert backend.interp_order == 2
    assert backend.min_mode_count == 8
    assert backend.interpolation_bound > backend.atol
    assert backend.sampling_bound <= backend.atol / 4


def test_tight_target_changes_geometry():
    """The tolerance affects interpolation preparation, not just tail windows."""
    loose, tight = FFT(512, atol=1e-6), FFT(512, atol=4e-15)
    assert not np.array_equal(loose.inverse_step, tight.inverse_step)
    assert tight.interpolation_bound < loose.interpolation_bound


def test_automatic_cutoff_fallback():
    """When no order meets the budget, enlarge the direct region and retry."""
    backend = FFT(64, min_mode_count=8, atol=4e-15)
    assert backend.min_mode_count > 8
    assert backend.interpolation_bound <= backend.atol / 4


@pytest.mark.parametrize("grad_atol", [None, 1e-7])
def test_report_reads_cached_bounds(monkeypatch, grad_atol):
    """Report access performs no planning or array-to-host conversion, even under JIT."""
    import drifft._backend.fft as module

    backend = FFT(
        256, grad_atol=grad_atol, angular_sensitivity=None if grad_atol is None else 1
    )

    def unexpected(*args, **kwargs):
        """Fail if reading the cached report invokes host planning again."""
        raise AssertionError("report recomputed geometry")

    monkeypatch.setattr(module, "_fft_grid", unexpected)
    monkeypatch.setattr(module, "_interpolation_bound", unexpected)
    import equinox as eqx

    report = eqx.filter_jit(lambda b: b.error_report)(backend)
    assert report.components[0].bound == backend.interpolation_bound
    assert report.components[1].bound == backend.sampling_bound
    assert report.components[0].grad_bound == backend.interpolation_grad_bound
    assert report.components[1].grad_bound == backend.sampling_grad_bound


@pytest.mark.parametrize("M", [32, 256, 512])
def test_direct_tail_and_reflection_bounds(M):
    """Sampling bounds cover direct truncation and both sides of reflection."""
    backend = FFT(M // 2)
    q = np.linspace(0.001, min(backend.min_mode_count / M, 0.5), 51)
    omitted = binom.pmf(np.arange(backend.boundary_width, M + 1), M, q[:, None])
    assert np.max(np.sum(omitted, axis=1)) <= backend.sampling_bound + 1e-15
    if not backend.buckets:
        low = binom.pmf(np.arange(M + 1), M, 0.5)
        low[backend.boundary_width :] = 0
        assert np.sum(abs(low - low[::-1])) <= 2 * backend.sampling_bound + 1e-15


def test_full_support_is_analytically_exact():
    """Small direct-only backends have no approximation error."""
    backend = FFT(8)
    assert backend.interpolation_bound == backend.sampling_bound == 0


@pytest.mark.parametrize("size", [16, 256, 512])
@pytest.mark.parametrize("sensitivity", [0.0, 0.1, 10.0])
def test_joint_gradient_planning(size, sensitivity):
    """One fixed geometry satisfies both analytic budgets in parameter units."""
    backend = FFT(size, atol=1e-8, grad_atol=1e-7, angular_sensitivity=sensitivity)
    report = backend.error_report
    assert report.grad_atol == 1e-7
    assert report.grad_coordinate is not None
    assert "branchwise" in report.grad_coordinate
    assert backend.interpolation_bound <= backend.atol / 4
    assert backend.sampling_bound <= backend.atol / 4
    assert backend.grad_atol is not None
    assert backend.interpolation_grad_bound is not None
    assert backend.sampling_grad_bound is not None
    assert backend.interpolation_grad_bound <= backend.grad_atol / 4
    assert backend.sampling_grad_bound <= backend.grad_atol / 4
    assert report.components[0].grad_bound == backend.interpolation_grad_bound
    assert report.components[1].grad_bound == backend.sampling_grad_bound
    assert report.grad_bound is None  # arithmetic remains uncertified
    if sensitivity == 0:
        value_only = FFT(size, atol=1e-8)
        assert backend.interp_order == value_only.interp_order
        assert backend.min_mode_count == value_only.min_mode_count
        assert backend.boundary_width == value_only.boundary_width
        assert backend.interpolation_grad_bound == backend.sampling_grad_bound == 0


def test_manual_order_cannot_silently_violate_gradient_target():
    """An explicit geometry conflicting with grad_atol must fail at construction."""
    with pytest.raises(ValueError, match="interp_order.*grad_atol"):
        FFT(
            256,
            interp_order=2,
            min_mode_count=8,
            grad_atol=1e-10,
            angular_sensitivity=1,
        )


def test_gradient_target_changes_geometry_and_tail_windows():
    """A gradient request must affect preparation, not merely attach a report."""
    base = FFT(512)
    joint = FFT(512, grad_atol=1e-10, angular_sensitivity=1)
    assert not np.array_equal(joint.inverse_step, base.inverse_step)
    assert joint.tail_tol < base.tail_tol
    assert joint.grad_atol is not None
    assert joint.interpolation_grad_bound is not None
    assert joint.sampling_grad_bound is not None
    assert (
        joint.interpolation_grad_bound + joint.sampling_grad_bound
        <= joint.grad_atol / 2
    )


@pytest.mark.parametrize(
    "grad_atol,sensitivity", [(1e-4, 1.0), (1e-7, 10.0), (1e-10, 1.0)]
)
def test_ideal_parameter_derivative_rows_near_switches(grad_atol, sensitivity):
    """Bound exact-node branch derivatives on both sides of knots and block edges."""
    M = 1024
    backend = FFT(M // 2, grad_atol=grad_atol, angular_sensitivity=sensitivity)
    assert backend.interpolation_grad_bound is not None
    assert backend.sampling_grad_bound is not None
    bound = backend.interpolation_grad_bound + backend.sampling_grad_bound
    cutoff = min(backend.min_mode_count / M, 0.5)
    boundaries = np.r_[cutoff, expit(np.asarray(backend.edges)), 1 - cutoff]
    for i in range(len(backend.eta0)):
        qmid = expit(float(backend.eta0[i]))
        queries = [
            np.nextafter(boundaries[i], boundaries[i + 1]),
            np.nextafter(boundaries[i + 1], boundaries[i]),
            np.nextafter(qmid, 0.0),
            qmid,
            np.nextafter(qmid, 1.0),
        ]
        for q in queries:
            if not boundaries[i] <= q <= boundaries[i + 1]:
                continue
            _, angular = _ideal_block(backend, i, q)
            exact = binom.pmf(np.arange(M + 1), M, q)
            derivative = exact * (np.arange(M + 1) - M * q) / np.sqrt(q * (1 - q))
            assert (
                np.sum(abs(sensitivity * (angular - derivative)))
                <= bound + sensitivity * 3e-10
            )


@pytest.mark.parametrize("width", [30, 50, 65, 101])
def test_direct_gradient_bound_against_omitted_rows(width):
    """Cover the tail-score formula, its interior maximum, and the CS fallback."""
    from drifft._backend.fft import _direct_grad_bound

    M, cutoff, sensitivity = 100, 0.5, 2.3
    tail = float(binom.sf(width - 1, M, cutoff))
    bound = _direct_grad_bound(M, width, cutoff, sensitivity, tail)
    q = np.linspace(1e-6, cutoff, 301)
    j = np.arange(width, M + 1)
    omitted = binom.pmf(j, M, q[:, None])
    derivative = (
        sensitivity * omitted * (j - M * q[:, None]) / np.sqrt(q * (1 - q))[:, None]
    )
    assert np.max(np.sum(abs(derivative), axis=1)) <= bound + 1e-11


@pytest.mark.parametrize(
    "grad_atol,sensitivity", [(1e-4, 1.0), (1e-7, 10.0), (1e-10, 1.0)]
)
def test_actual_joint_jax_derivatives(grad_atol, sensitivity):
    """JIT JVP/VJP and transpose gradients match independent binomial identities."""
    M = 512
    backend = FFT(M // 2, grad_atol=grad_atol, angular_sensitivity=sensitivity)
    cutoff = min(backend.min_mode_count / M, 0.5)
    switches = np.r_[cutoff, expit(np.asarray(backend.edges)), 1 - cutoff]
    q = jnp.asarray(
        np.unique(
            np.r_[
                0.0,
                0.01,
                0.381,
                0.513,
                0.99,
                1.0,
                np.nextafter(switches, 0.0),
                np.nextafter(switches, 1.0),
            ]
        )
    )
    tangent = sensitivity * jnp.sqrt(q * (1 - q))

    def rows(p, basis):
        """Expose coefficients to differentiate the complete operator row."""
        return jax.vmap(lambda v: backend.expect(p, v))(basis).T

    _, derivative = jax.jit(
        lambda p, basis, dp: jax.jvp(lambda x: rows(x, basis), (p,), (dp,))
    )(q, jnp.eye(M + 1), tangent)
    lower = binom.pmf(np.arange(M), M - 1, np.asarray(q)[:, None])
    reference = M * (np.pad(lower, ((0, 0), (1, 0))) - np.pad(lower, ((0, 0), (0, 1))))
    reference *= np.asarray(tangent)[:, None]
    # Finite-precision allowance is empirical and excluded from error_report.
    assert backend.interpolation_grad_bound is not None
    assert backend.sampling_grad_bound is not None
    budget = backend.interpolation_grad_bound + backend.sampling_grad_bound
    assert (
        np.max(np.sum(abs(derivative - reference), axis=1))
        <= budget + sensitivity * 2e-8
    )
    values = jnp.asarray(np.random.default_rng(19).uniform(-1, 1, M + 1))
    mass = jnp.linspace(-1, 1, len(q))
    gradient = jax.jit(jax.grad(lambda p: jnp.vdot(mass, backend.expect(p, values))))(q)
    transposed = jax.jit(
        jax.grad(lambda p: jnp.vdot(backend.advance(p, mass), values))
    )(q)
    expected = np.asarray(mass) * (reference @ values)
    np.testing.assert_allclose(
        gradient * tangent, expected, atol=budget + sensitivity * 2e-8, rtol=0
    )
    np.testing.assert_allclose(
        transposed * tangent, expected, atol=budget + sensitivity * 2e-8, rtol=0
    )


def test_direct_only_gradient_target_expands_window():
    """A tighter direct-tail derivative target retains additional binomial counts."""
    base = FFT(128)
    joint = FFT(128, grad_atol=1e-18, angular_sensitivity=1)
    assert joint.boundary_width > base.boundary_width
    assert not joint.buckets
    assert joint.grad_atol is not None
    assert joint.sampling_grad_bound is not None
    assert joint.sampling_grad_bound <= joint.grad_atol / 4
    j = np.arange(joint.boundary_width, 257)
    omitted_derivative = binom.pmf(j, 256, 0.5) * (j - 128) / 0.5
    assert np.sum(abs(omitted_derivative)) <= joint.sampling_grad_bound * (1 + 1e-10)


def test_unrepresentable_gradient_budget_fails():
    """Underflow cannot turn an impossible budget into an apparent success."""
    with pytest.raises(ValueError, match="grad_atol.*too small"):
        FFT(16, grad_atol=float(np.nextafter(0.0, 1.0)), angular_sensitivity=1)


def test_feasible_manual_order_gradient_request():
    """Explicit order remains usable when its analytic derivative bound fits."""
    backend = FFT(256, interp_order=12, grad_atol=1e-4, angular_sensitivity=1)
    assert backend.interp_order == 12
    assert backend.interpolation_grad_bound is not None
    assert backend.sampling_grad_bound is not None
    assert backend.interpolation_grad_bound + backend.sampling_grad_bound <= 1e-4


@pytest.mark.parametrize(
    "order,grad_atol", [(4, 0.1), (8, 1e-4), (12, 1e-6), (16, 1e-9)]
)
def test_manual_derivative_bounds_against_ideal_interpolants(order, grad_atol):
    """Exercise non-negligible interpolation errors, not only roundoff-sized ones."""
    M, sensitivity = 512, 1.7
    backend = FFT(
        M // 2, interp_order=order, grad_atol=grad_atol, angular_sensitivity=sensitivity
    )
    assert backend.interpolation_grad_bound is not None
    assert backend.sampling_grad_bound is not None
    bound = backend.interpolation_grad_bound + backend.sampling_grad_bound
    cutoff = backend.min_mode_count / M
    edges = np.r_[cutoff, expit(np.asarray(backend.edges)), 1 - cutoff]
    for i in range(len(backend.eta0)):
        for q in np.linspace(edges[i], edges[i + 1], 7)[1:-1]:
            _, angular = _ideal_block(backend, i, q)
            lower = binom.pmf(np.arange(M), M - 1, q)
            expected = (
                M * np.sqrt(q * (1 - q)) * (np.r_[0.0, lower] - np.r_[lower, 0.0])
            )
            assert np.sum(abs(sensitivity * (angular - expected))) <= bound + 1e-11


def test_fft_planning_is_independent_of_decimal_context():
    """Caller rounding/trap settings must not leak into host preparation."""
    from decimal import ROUND_DOWN, Inexact, localcontext

    from drifft._backend.fft import _interpolation_bound

    expected = FFT(257, grad_atol=1e-7, angular_sensitivity=1)
    # Avoid a cache hit hiding a dependency on the ambient Decimal context.
    _interpolation_bound.cache_clear()
    with localcontext() as context:
        context.prec = 5
        context.rounding = ROUND_DOWN
        context.traps[Inexact] = True
        actual = FFT(257, grad_atol=1e-7, angular_sensitivity=1)
        assert context.prec == 5
        assert context.rounding == ROUND_DOWN
        assert context.traps[Inexact]
    assert actual.error_report == expected.error_report
    assert actual.interp_order == expected.interp_order
    assert actual.min_mode_count == expected.min_mode_count
    np.testing.assert_array_equal(actual.rmin, expected.rmin)


@pytest.mark.parametrize("tail_function,pop_size", [("cdf", 256), ("sf", 64)])
def test_nonfinite_tail_evaluation_fails_closed(monkeypatch, tail_function, pop_size):
    """A numerical tail failure cannot be mistaken for zero omitted mass."""
    monkeypatch.setattr(binom, tail_function, lambda *args, **kwargs: float("nan"))
    with pytest.raises(ValueError, match="tail"):
        FFT(pop_size)
